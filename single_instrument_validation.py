"""
single_instrument_validation.py — Passo 4 dell'imbuto di selezione
asset: validazione statistica del segnale Variante 6 su un singolo
strumento, z-score contro baseline random (30 seed), stesso metodo
esatto usato per la generalizzazione originale DAX/FTSE100 (RCA sez.5,
12/07) e per NIKKEI225 (RCA Addendum 13/07 sez.20).

Diverso da asset_pair_comparison.py (quello confronta coppie sul
PnL/drawdown già DENTRO l'universo attivo) — qui la domanda è più a
monte: "il segnale ha un vantaggio statistico reale su questo
strumento, isolato, prima ancora di pensare a quale coppia formare?"

Diverso anche da ema_grid_search.run_portfolio/random_baseline_pnls
(quelle funzioni usano eng.BacktestEngine con INSTRUMENTS di default
hardcoded — non includerebbero un simbolo nuovo come SMI). Qui il
motore viene istanziato esplicitamente con lo strumento richiesto.

Uso: python single_instrument_validation.py SMI

Per aggiungere un nuovo strumento: aggiungere una riga a
INSTRUMENT_REGISTRY con i parametri verificati (screenshot IG Get Info
per size/margine/spread, ATR/lookback INIZIALMENTE copiati da
DAX/FTSE100 come primo passaggio senza calibrazione — se il segnale
grezzo mostra comunque vantaggio, poi si passa alla calibrazione vera
come già fatto per GOLD in RCA sez.22.3; se il vantaggio grezzo non
c'è nemmeno, il segnale si ferma qui senza sprecare una calibrazione
completa, stesso principio dell'imbuto).

NOTA SU SMI: spread AGGIORNATO il 15/07/2026 con osservazione in
orario di mercato attivo (SELL 14199.4/BUY 14201.4 = 2.0 punti),
sostituisce la prima stima weekend (6.0 punti, gonfiata — stessa
trappola già documentata più volte nel progetto, RCA sez.1/13). ATR
multiplier e lookback restano COPIATI da DAX, non calibrati — se
questo giro con spread corretto mostra vantaggio ma PnL assoluto
ancora debole, il prossimo passo è la calibrazione vera (grid search),
non un'altra correzione di singolo parametro.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import engine as eng
import ema_grid_search as g

CAPITAL0 = 2000.0
TRAIN_RANDOM_SEEDS = 15
VALIDATION_RANDOM_SEEDS = 30
ALL_PERIODS = list(g.PERIODS.keys())

INSTRUMENT_REGISTRY: dict[str, eng.InstrumentConfig] = {
    "SMI": eng.InstrumentConfig(
        name="SMI", tradable=True,
        breakout_lookback=20, atr_multiplier=1.5,   # COPIATI da DAX, non calibrati
        risk_pct=0.015,                              # allineato a FTSE100 (candidato sostituto)
        point_value=2.16,                            # EUR stimato da CHF 2 (screenshot 15/07)
        spread_fixed=2.0,                             # punti, verificato in orario di mercato attivo (15/07/2026, SELL 14199.4/BUY 14201.4)
        min_tradable_size=0.10, margin_pct=0.10,
    ),
}


def make_random_signal_df(signal_df: pd.DataFrame, n_long: int, n_short: int,
                           rng: np.random.Generator) -> pd.DataFrame:
    df = signal_df.copy()
    df["signal"] = None
    eligible = df.index[df["atr"].notna() & df["adx"].notna()]
    eligible = eligible[eligible < len(df) - 1]
    n_needed = n_long + n_short
    if len(eligible) < n_needed:
        n_needed = len(eligible)
        n_long = min(n_long, n_needed)
        n_short = n_needed - n_long
    chosen = rng.choice(eligible, size=n_needed, replace=False)
    rng.shuffle(chosen)
    df.loc[chosen[:n_long], "signal"] = "long"
    df.loc[chosen[n_long:n_long + n_short], "signal"] = "short"
    return df


def count_real_signals(signal_df: pd.DataFrame) -> tuple[int, int]:
    n_long = int((signal_df["signal"] == "long").sum())
    n_short = int((signal_df["signal"] == "short").sum())
    return n_long, n_short


def zscore(real_pnl: float, random_pnls: list[float]) -> float:
    arr = np.array(random_pnls)
    std = arr.std(ddof=1)
    return 0.0 if std == 0 else (real_pnl - arr.mean()) / std


def run_single(symbol: str, inst: eng.InstrumentConfig, sig_df: pd.DataFrame,
                capital0: float) -> tuple[float, int, float]:
    """Ritorna (pnl_total, num_trades, max_drawdown_pct)."""
    engine_ = eng.BacktestEngine(capital0=capital0, instruments={symbol: inst})
    trades_df, metrics_df = engine_.run({symbol: sig_df})
    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return pnl, n, dd


def validate_symbol(symbol: str, inst: eng.InstrumentConfig, capital0: float = CAPITAL0):
    full_df = g.load_full_ohlc(f"{symbol}_full.csv")
    rows = []

    for period in ALL_PERIODS:
        window, period_start = g.slice_period(full_df, period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)

        real_pnl, n_trades, real_dd = run_single(symbol, inst, sig, capital0)

        if n_trades == 0:
            rows.append({"period": period, "num_trades": 0, "pnl_total": 0.0,
                         "max_drawdown_pct": 0.0, "z_score": np.nan})
            print(f"  [{period}] 0 trade, salto z-score.")
            continue

        n_long, n_short = count_real_signals(sig)
        n_seeds = VALIDATION_RANDOM_SEEDS
        random_pnls = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            rand_sig = make_random_signal_df(sig, n_long, n_short, rng)
            r_pnl, _, _ = run_single(symbol, inst, rand_sig, capital0)
            random_pnls.append(r_pnl)

        z = zscore(real_pnl, random_pnls)
        rows.append({"period": period, "num_trades": n_trades, "pnl_total": real_pnl,
                     "max_drawdown_pct": real_dd, "z_score": z,
                     "random_pnl_mean": float(np.mean(random_pnls))})
        print(f"  [{period}] trades={n_trades} pnl={real_pnl:.1f} "
              f"dd={real_dd*100:.2f}% z={z:.2f} (random_mean={np.mean(random_pnls):.1f})")

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Uso: python single_instrument_validation.py SIMBOLO")
        print(f"Simboli disponibili: {', '.join(INSTRUMENT_REGISTRY.keys())}")
        sys.exit(1)

    symbol = sys.argv[1].strip().upper()
    if symbol not in INSTRUMENT_REGISTRY:
        print(f"ERRORE: {symbol} non in INSTRUMENT_REGISTRY. "
              f"Disponibili: {', '.join(INSTRUMENT_REGISTRY.keys())}")
        sys.exit(1)

    inst = INSTRUMENT_REGISTRY[symbol]
    print(f"=== Validazione standalone {symbol} (capitale {CAPITAL0}€) ===")
    print(f"Parametri: lookback={inst.breakout_lookback} ATR×{inst.atr_multiplier} "
          f"risk={inst.risk_pct*100}% spread={inst.spread_fixed} min_size={inst.min_tradable_size}")

    df_results = validate_symbol(symbol, inst)

    import os
    os.makedirs("results", exist_ok=True)
    out_path = f"results/single_validation_{symbol.lower()}.csv"
    df_results.to_csv(out_path, index=False)

    valid = df_results.dropna(subset=["z_score"])
    n_positive_periods = int((valid["z_score"] > 0).sum())
    sum_z = valid["z_score"].sum()

    print(f"\n=== RIEPILOGO {symbol} ===")
    print(df_results.to_string(index=False))
    print(f"\nPeriodi con z-score positivo: {n_positive_periods}/{len(valid)}")
    print(f"Somma z-score sui periodi validi: {sum_z:.2f}")
    print(f"\n(Riferimento noti: NIKKEI225 somma z=14.50, 5/5 periodi positivi — "
          f"segnale eccellente. DAX 2024-25 singolo periodo z=+4.16 — molto forte.)")
    print(f"\nCompletato. File: {out_path}")


if __name__ == "__main__":
    main()
