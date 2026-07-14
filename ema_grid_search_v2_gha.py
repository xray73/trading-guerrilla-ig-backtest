"""
ema_grid_search.py — Grid search EMA (coppia veloce + coppia ampia)
Agente AI Trading Guerrilla IG — Fase 1 backtest offline
=====================================================================

Da eseguire in Google Colab, nella STESSA cartella dove hai engine.py
(quello scaricato/allineato al repo GitHub, versione 13/07/2026, v2).

Cosa fa:
  1. Carica OHLC DAX+FTSE100 da CSV (esportati da D1 via wrangler, vedi
     istruzioni in fondo al file / messaggio chat).
  2. Costruisce una griglia 5x5 di coppie EMA (veloce + ampia), 20 combo +
     1 baseline (20/50 + 100/200) = 25 combo totali. Lookback breakout
     (20 DAX / 40 FTSE100), ATR 1.5x, rischio 2.0%/1.5% RESTANO FISSI —
     isoliamo solo l'effetto delle EMA.
  3. Disciplina walk-forward (lezione da RCA Addendum sez.17 — filtro
     maturità bocciato per overfitting da multiple testing):
        - TRAIN: seleziona il/i migliori candidati SOLO su 2023
        - TEST:  verifica SOLO su 2024-25, nessun ritocco parametri
        - CONFERMA: solo se il TEST è superato, sui 3 periodi restanti
          (2015-2016, 2020-covid, 2026-ytd)
  4. Baseline random: 30 seed per combo/periodo, stesso numero di trade
     del segnale reale (portfolio DAX+FTSE100 insieme, stesse regole di
     uscita/risk management — Charter sez.6), z-score = (pnl_reale -
     media_random) / std_random.
  5. Output: SOLO metriche aggregate in CSV su Drive (mai trade
     individuali in chat, per Regole_Backtest_MonteCarlo.md).

Nota onestà metodologica: con 25 combinazioni testate in train, il
rischio di multiple-testing overfitting è reale (stesso problema del
filtro maturità, Addendum sez.17). Per questo la selezione in train usa
un margine minimo esplicito sopra il baseline (non "il migliore qualunque
sia il vantaggio") e la verifica out-of-sample è vincolante: se il
vantaggio non regge su 2024-25, si resta sul baseline 20/50+100/200,
punto. Nessuna nuova soglia si abbassa per far passare un risultato.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import numpy as np
import pandas as pd
from pathlib import Path

import engine as eng  # engine.py deve essere nella stessa cartella / sys.path


# =====================================================================
# 0. CONFIGURAZIONE GRID SEARCH
# =====================================================================

FAST_PAIRS = [(10, 30), (15, 40), (20, 50), (25, 60), (30, 70)]   # (fast, slow)
BROAD_PAIRS = [(75, 150), (100, 200), (125, 250), (150, 300), (175, 350)]

BASELINE_FAST = (20, 50)
BASELINE_BROAD = (100, 200)

TRAIN_RANDOM_SEEDS = 15        # screening iniziale, 25 combo — ridotto per
                                # tempo (precedente: RCA sez.5 usa 10 seed
                                # per esplorazione, dichiarato "indicativo,
                                # non della stessa robustezza statistica")
VALIDATION_RANDOM_SEEDS = 30   # test + conferma, solo sui pochi candidati
                                # promossi — piena rigorosità (RCA sez.1, 11)
Z_MARGIN_TRAIN = 1.0           # margine minimo di z-score sopra il baseline
                                # per essere promosso a candidato (evita di
                                # promuovere rumore da 25 combo testate)
CAPITAL0 = 900.0               # capitale di riferimento simulazione (Charter sez.3)

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),          # TRAIN
    "2024-2025": ("2024-01-01", "2025-12-31"),      # TEST (out-of-sample)
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}
TRAIN_PERIOD = "2023"
TEST_PERIOD = "2024-2025"
CONFIRM_PERIODS = ["2015-2016", "2020-covid", "2026-ytd"]

WARMUP_DAYS = 90  # buffer prima di period_start per stabilizzare EMA fino a 350


# =====================================================================
# 1. CARICAMENTO DATI (CSV esportati da D1 via wrangler, vedi istruzioni)
# =====================================================================

def load_full_ohlc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    start_str, end_str = PERIODS[period_label]
    start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return window, pd.Timestamp(start_str, tz="UTC")


# =====================================================================
# 2. SIGNAL GENERATION CON EMA CUSTOM (usa engine.py, override parametri)
# =====================================================================

def generate_signals_custom(raw_df: pd.DataFrame, inst: eng.InstrumentConfig,
                             fast_pair: tuple[int, int], broad_pair: tuple[int, int]
                             ) -> pd.DataFrame:
    p = dataclasses.replace(
        eng.PARAMS,
        ema_fast=fast_pair[0], ema_slow=fast_pair[1],
        ema_broad_fast=broad_pair[0], ema_broad_slow=broad_pair[1],
    )
    return eng.generate_signals(raw_df, inst, p)


def trim_warmup(df: pd.DataFrame, period_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= period_start].reset_index(drop=True)


# =====================================================================
# 3. BACKTEST PORTFOLIO (DAX+FTSE100 insieme, come in produzione)
# =====================================================================

def run_portfolio(data: dict[str, pd.DataFrame], capital0: float = CAPITAL0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine_ = eng.BacktestEngine(capital0=capital0)
    trades_df, metrics_df = engine_.run(data)
    return trades_df, metrics_df


# =====================================================================
# 4. BASELINE RANDOM — stesso numero di trade, stesse regole di uscita
# =====================================================================

def make_random_signal_df(signal_df: pd.DataFrame, n_long: int, n_short: int,
                           rng: np.random.Generator) -> pd.DataFrame:
    df = signal_df.copy()
    df["signal"] = None
    eligible = df.index[df["atr"].notna() & df["adx"].notna()]
    eligible = eligible[eligible < len(df) - 1]  # serve la barra N+1 per entrare
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


def random_baseline_pnls(data_real: dict[str, pd.DataFrame],
                          real_signal_counts: dict[str, tuple[int, int]],
                          n_seeds: int = TRAIN_RANDOM_SEEDS, capital0: float = CAPITAL0
                          ) -> list[float]:
    """real_signal_counts: {instrument: (n_long, n_short)} dal run reale."""
    pnls = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        data_rand = {}
        for name, df in data_real.items():
            n_long, n_short = real_signal_counts.get(name, (0, 0))
            data_rand[name] = make_random_signal_df(df, n_long, n_short, rng)
        _, metrics = run_portfolio(data_rand, capital0)
        pnls.append(float(metrics["pnl_total"].iloc[0]))
    return pnls


def count_real_signals(signal_df: pd.DataFrame) -> tuple[int, int]:
    n_long = int((signal_df["signal"] == "long").sum())
    n_short = int((signal_df["signal"] == "short").sum())
    return n_long, n_short


def zscore(real_pnl: float, random_pnls: list[float]) -> float:
    arr = np.array(random_pnls)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return (real_pnl - arr.mean()) / std


# =====================================================================
# 5. EVAL DI UNA COMBINAZIONE SU UN PERIODO
# =====================================================================

def eval_combo_on_period(fast_pair, broad_pair, period_label: str,
                          full_data: dict[str, pd.DataFrame],
                          n_seeds: int = TRAIN_RANDOM_SEEDS) -> dict:
    data_real = {}
    for name, full_df in full_data.items():
        inst = eng.INSTRUMENTS[name]
        window, period_start = slice_period(full_df, period_label)
        sig = generate_signals_custom(window, inst, fast_pair, broad_pair)
        sig = trim_warmup(sig, period_start)
        data_real[name] = sig

    trades_df, metrics_df = run_portfolio(data_real)
    real_pnl = float(metrics_df["pnl_total"].iloc[0])
    num_trades = int(metrics_df["num_trades"].iloc[0])

    signal_counts = {name: count_real_signals(df) for name, df in data_real.items()}
    random_pnls = random_baseline_pnls(data_real, signal_counts, n_seeds=n_seeds)
    z = zscore(real_pnl, random_pnls)

    return {
        "fast_pair": f"{fast_pair[0]}/{fast_pair[1]}",
        "broad_pair": f"{broad_pair[0]}/{broad_pair[1]}",
        "period": period_label,
        "num_trades": num_trades,
        "pnl_total": real_pnl,
        "win_rate": float(metrics_df["win_rate"].iloc[0]) if num_trades else np.nan,
        "profit_factor": float(metrics_df["profit_factor"].iloc[0]) if num_trades else np.nan,
        "max_drawdown_pct": float(metrics_df["max_drawdown_pct"].iloc[0]) if num_trades else np.nan,
        "z_score": z,
        "random_pnl_mean": float(np.mean(random_pnls)),
        "random_pnl_std": float(np.std(random_pnls, ddof=1)),
    }


# =====================================================================
# 6. MAIN — TRAIN -> SELEZIONE -> TEST -> CONFERMA
# =====================================================================

def main(dax_csv: str, ftse_csv: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    full_data = {
        "DAX": load_full_ohlc(dax_csv),
        "FTSE100": load_full_ohlc(ftse_csv),
    }

    combos = list(itertools.product(FAST_PAIRS, BROAD_PAIRS))
    print(f"[TRAIN] {len(combos)} combinazioni su periodo {TRAIN_PERIOD} "
          f"({TRAIN_RANDOM_SEEDS} seed random per screening)...")

    train_rows = []
    for i, (fast, broad) in enumerate(combos, 1):
        row = eval_combo_on_period(fast, broad, TRAIN_PERIOD, full_data,
                                    n_seeds=TRAIN_RANDOM_SEEDS)
        train_rows.append(row)
        tag = " [BASELINE]" if (fast, broad) == (BASELINE_FAST, BASELINE_BROAD) else ""
        print(f"  {i}/{len(combos)} fast={row['fast_pair']} broad={row['broad_pair']} "
              f"z={row['z_score']:.2f} trades={row['num_trades']} pnl={row['pnl_total']:.0f}{tag}")

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(f"{output_dir}/ema_grid_train_2023.csv", index=False)

    baseline_row = train_df[
        (train_df["fast_pair"] == f"{BASELINE_FAST[0]}/{BASELINE_FAST[1]}") &
        (train_df["broad_pair"] == f"{BASELINE_BROAD[0]}/{BASELINE_BROAD[1]}")
    ].iloc[0]
    baseline_z = baseline_row["z_score"]
    print(f"\n[BASELINE 20/50+100/200] z_train (screening, {TRAIN_RANDOM_SEEDS} seed) = {baseline_z:.2f}")

    # candidati: z_score sopra baseline + margine esplicito (fissato PRIMA di vedere i risultati)
    candidates = train_df[train_df["z_score"] >= baseline_z + Z_MARGIN_TRAIN].sort_values(
        "z_score", ascending=False)

    if candidates.empty:
        print(f"\nNESSUN candidato supera il baseline di margine {Z_MARGIN_TRAIN} in train. "
              f"Si resta su 20/50+100/200 — grid search chiusa qui, nessuna verifica out-of-sample necessaria.")
        return

    print(f"\n[CANDIDATI] {len(candidates)} combinazioni promosse al TEST (2024-2025), "
          f"ora con {VALIDATION_RANDOM_SEEDS} seed pieni:")
    print(candidates[["fast_pair", "broad_pair", "z_score", "pnl_total"]].to_string(index=False))

    # TEST — nessun ritocco, verifica pura, piena rigorosità statistica
    test_rows = []
    for _, c in candidates.iterrows():
        fast = tuple(int(x) for x in c["fast_pair"].split("/"))
        broad = tuple(int(x) for x in c["broad_pair"].split("/"))
        row = eval_combo_on_period(fast, broad, TEST_PERIOD, full_data,
                                    n_seeds=VALIDATION_RANDOM_SEEDS)
        test_rows.append(row)
        print(f"  [TEST] fast={row['fast_pair']} broad={row['broad_pair']} "
              f"z={row['z_score']:.2f} pnl={row['pnl_total']:.0f}")

    test_df = pd.DataFrame(test_rows)
    baseline_test = eval_combo_on_period(BASELINE_FAST, BASELINE_BROAD, TEST_PERIOD, full_data,
                                          n_seeds=VALIDATION_RANDOM_SEEDS)
    print(f"\n[BASELINE 20/50+100/200] z_test = {baseline_test['z_score']:.2f} "
          f"pnl={baseline_test['pnl_total']:.0f}")

    test_df["baseline_z_test"] = baseline_test["z_score"]
    test_df["beats_baseline_test"] = test_df["z_score"] > baseline_test["z_score"]
    test_df.to_csv(f"{output_dir}/ema_grid_test_2024_2025.csv", index=False)

    survivors = test_df[test_df["beats_baseline_test"]].sort_values("z_score", ascending=False)

    if survivors.empty:
        print("\nNESSUN candidato batte il baseline fuori campione (2024-2025). "
              "Si resta su 20/50+100/200 — pattern train-vince/test-crolla, "
              "stesso esito del filtro maturità (Addendum sez.17). Chiuso.")
        return

    print(f"\n[SOPRAVVISSUTI] {len(survivors)} combinazioni passano alla CONFERMA "
          f"sui 3 periodi restanti:")
    print(survivors[["fast_pair", "broad_pair", "z_score"]].to_string(index=False))

    confirm_rows = []
    for _, c in survivors.iterrows():
        fast = tuple(int(x) for x in c["fast_pair"].split("/"))
        broad = tuple(int(x) for x in c["broad_pair"].split("/"))
        for period in CONFIRM_PERIODS:
            row = eval_combo_on_period(fast, broad, period, full_data,
                                        n_seeds=VALIDATION_RANDOM_SEEDS)
            confirm_rows.append(row)
            print(f"  [CONFERMA {period}] fast={row['fast_pair']} broad={row['broad_pair']} "
                  f"z={row['z_score']:.2f} pnl={row['pnl_total']:.0f}")

    confirm_df = pd.DataFrame(confirm_rows)
    confirm_df.to_csv(f"{output_dir}/ema_grid_confirm_3periods.csv", index=False)

    print("\n=== RIEPILOGO FINALE (somma z-score su 5 periodi, per combo sopravvissuta) ===")
    for _, c in survivors.iterrows():
        fast, broad = c["fast_pair"], c["broad_pair"]
        z_train = c["z_score"]
        z_test = test_df.loc[
            (test_df["fast_pair"] == fast) & (test_df["broad_pair"] == broad), "z_score"
        ].iloc[0]
        z_confirm_sum = confirm_df.loc[
            (confirm_df["fast_pair"] == fast) & (confirm_df["broad_pair"] == broad), "z_score"
        ].sum()
        z_total = z_train + z_test + z_confirm_sum
        print(f"fast={fast} broad={broad}: z_totale_5periodi={z_total:.2f} "
              f"(train={z_train:.2f}, test={z_test:.2f}, confirm_sum={z_confirm_sum:.2f})")

    print("\nConfronta z_totale con quello del baseline attuale (somma nota da RCA: "
          "DAX 12.94 + FTSE100 8.53 = 21.47 sui 5 periodi, MA quella è somma per-strumento, "
          "qui è portfolio — ricalcola baseline sui 5 periodi con lo stesso script per confronto "
          "omogeneo prima di decidere un cambio parametro operativo).")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python ema_grid_search.py DAX_full.csv FTSE100_full.csv [output_dir]")
        sys.exit(1)
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    main(sys.argv[1], sys.argv[2], out_dir)
