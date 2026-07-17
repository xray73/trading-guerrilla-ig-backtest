"""
spread_sensitivity_revalidation.py — Test di sensibilità (17/07/2026,
NON una ricalibrazione definitiva — solo 2-3 campioni spread raccolti
finora, troppo pochi per essere conclusivi): quanto cambia l'edge di
Variante 6 sui 5 periodi ufficiali se lo spread fosse quello osservato
su IG in orario di mercato invece di quello assunto originariamente
nel motore (mai verificato empiricamente fino ad oggi)?

Confronto:
  ORIGINALE:  spread_fixed = DAX 1.2pt, FTSE100 1.0pt (valori attuali
              in engine.py, mai verificati)
  REALISTICO: spread_fixed = DAX 2.5pt, FTSE100 1.5pt (media dei
              campioni IG raccolti il 17/07/2026 — n=2 per strumento,
              campione MOLTO piccolo, da NON trattare come definitivo)

Nessuna modifica a engine.py — i parametri modificati sono passati
come istanza locale via dataclasses.replace(). generate_signals()
resta INVARIATO in entrambi gli scenari (lo spread non influenza la
generazione del segnale, solo il prezzo di entrata/uscita — coerente
con come è strutturato engine.py).

Ogni periodo è indipendente (capitale riparte da CAPITAL0), coerente
con la metodologia walk-forward del progetto.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

# spread realistico basato sui campioni IG raccolti il 17/07/2026 (n=2-3
# per strumento — campione piccolo, da rifare con più dati quando possibile)
REALISTIC_SPREAD = {"DAX": 2.5, "FTSE100": 1.5}


def fetch_bars(symbol_const, start, end) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_signal_data(period_start: str, period_end: str) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    signal_data = {}
    for name, const in SYMBOLS.items():
        raw = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        inst = eng.INSTRUMENTS[name]
        full_signals = eng.generate_signals(raw, inst)
        signal_data[name] = full_signals[full_signals["timestamp"] >= p_start].reset_index(drop=True)
    return signal_data


def metrics_summary(trades_df: pd.DataFrame, instrument: str | None = None) -> dict:
    sub = trades_df if instrument is None else trades_df[trades_df["instrument"] == instrument]
    n = len(sub)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "pnl_total": 0.0}
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": sub["pnl"].sum()}


def main():
    print("=== Test di sensibilità spread — Variante 6, 5 periodi ufficiali ===")
    print("ATTENZIONE: spread realistico basato su campione MOLTO piccolo (n=2-3/strumento).")
    print("Trattare come indicazione preliminare, non come ricalibrazione definitiva.\n")

    original_instruments = eng.INSTRUMENTS
    realistic_instruments = {
        name: dataclasses.replace(inst, spread_fixed=REALISTIC_SPREAD[name])
        for name, inst in eng.INSTRUMENTS.items()
    }

    all_rows = []
    for label, p_start, p_end in PERIODS:
        print(f"\n--- Periodo {label} ---")
        signal_data = get_period_signal_data(p_start, p_end)

        engine_orig = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=original_instruments)
        trades_orig, _ = engine_orig.run(signal_data)

        engine_real = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=realistic_instruments)
        trades_real, _ = engine_real.run(signal_data)

        m_orig = metrics_summary(trades_orig)
        m_real = metrics_summary(trades_real)

        pnl_delta = m_real["pnl_total"] - m_orig["pnl_total"]
        pnl_delta_pct = 100 * pnl_delta / abs(m_orig["pnl_total"]) if m_orig["pnl_total"] != 0 else np.nan

        print(f"  Spread originale (1.2/1.0): n={m_orig['n_trades']} WR={m_orig['win_rate_pct']:.1f}% "
              f"PF={m_orig['profit_factor']:.2f} PnL={m_orig['pnl_total']:+.2f}")
        print(f"  Spread realistico (2.5/1.5): n={m_real['n_trades']} WR={m_real['win_rate_pct']:.1f}% "
              f"PF={m_real['profit_factor']:.2f} PnL={m_real['pnl_total']:+.2f}")
        print(f"  Impatto: {pnl_delta:+.2f} EUR ({pnl_delta_pct:+.1f}%)")

        all_rows.append({
            "periodo": label,
            "orig_n_trades": m_orig["n_trades"], "orig_wr_pct": m_orig["win_rate_pct"],
            "orig_pf": m_orig["profit_factor"], "orig_pnl": m_orig["pnl_total"],
            "real_n_trades": m_real["n_trades"], "real_wr_pct": m_real["win_rate_pct"],
            "real_pf": m_real["profit_factor"], "real_pnl": m_real["pnl_total"],
            "pnl_delta": pnl_delta, "pnl_delta_pct": pnl_delta_pct,
        })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("spread_sensitivity_revalidation.csv", index=False)

    avg_pf_orig = summary_df["orig_pf"].mean()
    avg_pf_real = summary_df["real_pf"].mean()
    total_pnl_orig = summary_df["orig_pnl"].sum()
    total_pnl_real = summary_df["real_pnl"].sum()

    print(f"\n{'='*60}\nRIEPILOGO\n{'='*60}")
    print(f"PF medio — originale: {avg_pf_orig:.2f}  realistico: {avg_pf_real:.2f}")
    print(f"PnL totale (5 periodi) — originale: {total_pnl_orig:+.2f}  realistico: {total_pnl_real:+.2f}")
    print(f"Periodi ancora positivi con spread realistico: "
          f"{(summary_df['real_pnl'] > 0).sum()}/5")
    print("\nFile: spread_sensitivity_revalidation.csv")


if __name__ == "__main__":
    main()
