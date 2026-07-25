"""
run_official_backtest_5periods.py — Riesecuzione ufficiale di Variante 6
sui 5 periodi standard dopo l'aggiornamento spread del 24/07/2026
(engine.py: spread_fixed DAX 1.2->2.68pt, FTSE100 1.0->1.57pt, media
reale osservata su spread_samples D1, n=108/strumento).

Motore usato: BacktestEngineFloatingKillSwitch (adottato in produzione,
vedi decision log 15-19/07/2026 — kill switch su floating loss, non solo
realizzato). Capitale 2.000€ (vincolo reale, non i 900€ di riferimento
usati in alcune analisi esplorative precedenti).

Riusa le utility già scritte e testate in ema_grid_search.py (PERIODS,
load_full_ohlc, slice_period, trim_warmup) — stessa definizione dei 5
periodi ufficiali già in uso in floating_kill_switch_impact.py e altri
script di produzione, nessuna reimplementazione.

Output:
  - results/official_trades_<periodo>.csv   (trade individuali, per periodo)
  - results/official_metrics_5periods.csv   (metriche per periodo + aggregato)
Le metriche aggregate vengono anche stampate a video (run_metrics, ok da
condividere in chat per Regole_Backtest_MonteCarlo.md). I trade individuali
restano nell'artifact — NON vanno incollati in chat.

NOTA: questo script NON scrive su D1 (backtest_runs/trades/run_metrics).
La funzione export_trades_for_d1() in engine.py ha un bug noto: non
popola la colonna stop_loss, che è NOT NULL nello schema reale (verificato
via PRAGMA table_info() il 25/07/2026) — va corretta prima di un eventuale
upload dei trade su D1. Qui i risultati restano solo su CSV/artifact.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0  # capitale reale (Charter, non i 900€ di riferimento esplorativo)


def load_full_ohlc_mixed(csv_path: str) -> pd.DataFrame:
    """Sostituisce g.load_full_ohlc(): quella usa pd.to_datetime senza
    format='mixed', e va in crash su DAX_full.csv/FTSE100_full.csv perché
    ohlc_prices in D1 contiene sia righe storiche (formato con spazio,
    "2026-07-10 19:30:00+00:00") sia righe scritte con isoformat() prima
    del fix del 20/07/2026 in ohlc_data_source.py (formato con T,
    "2026-07-12T22:00:00+00:00") — stesso bug, stesso fix, qui applicato
    localmente per non modificare ema_grid_search.py (usato da altri
    script già funzionanti su dataset senza righe in formato T)."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df.sort_values("timestamp").reset_index(drop=True)


def run_period(period_label: str, full_data: dict) -> tuple[pd.DataFrame, dict]:
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period_label)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig

    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_df, metrics_df = engine_.run(data)

    n = int(metrics_df["num_trades"].iloc[0])
    pnl = float(metrics_df["pnl_total"].iloc[0])
    wr = float(metrics_df["win_rate"].iloc[0]) if n else np.nan
    pf_raw = metrics_df["profit_factor"].iloc[0]
    pf = float(pf_raw) if n and np.isfinite(pf_raw) else np.nan
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0

    trades_df = trades_df.copy()
    trades_df["periodo"] = period_label

    return trades_df, {
        "periodo": period_label,
        "num_trades": n,
        "win_rate_pct": wr * 100 if n else np.nan,
        "profit_factor": pf,
        "pnl_total": pnl,
        "max_drawdown_pct": dd * 100,
        "capital_final": float(metrics_df["capital_final"].iloc[0]),
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("=== Backtest ufficiale V6, 5 periodi — spread reale aggiornato ===")
    print(f"engine.py spread_fixed: DAX={eng.INSTRUMENTS['DAX'].spread_fixed}pt, "
          f"FTSE100={eng.INSTRUMENTS['FTSE100'].spread_fixed}pt")
    print(f"Capitale iniziale per periodo: {CAPITAL0:.0f}€ (walk-forward, ogni periodo riparte da qui)\n")

    full_data = {
        "DAX": load_full_ohlc_mixed("DAX_full.csv"),
        "FTSE100": load_full_ohlc_mixed("FTSE100_full.csv"),
    }

    all_trades = []
    metrics_rows = []

    for period_label in g.PERIODS:
        trades_df, row = run_period(period_label, full_data)
        metrics_rows.append(row)
        all_trades.append(trades_df)

        trades_df.to_csv(f"results/official_trades_{period_label}.csv", index=False)

        pf_str = f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "n/a"
        print(f"[{period_label:12s}] n={row['num_trades']:4d}  WR={row['win_rate_pct']:5.1f}%  "
              f"PF={pf_str:>5s}  PnL={row['pnl_total']:+9.2f}€  "
              f"DD={row['max_drawdown_pct']:6.2f}%  capitale_finale={row['capital_final']:.2f}€")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("results/official_metrics_5periods.csv", index=False)

    print(f"\n{'='*78}\nRIEPILOGO AGGREGATO (5 periodi ufficiali, spread reale)\n{'='*78}")
    total_trades = int(metrics_df["num_trades"].sum())
    total_pnl = float(metrics_df["pnl_total"].sum())
    avg_pf = float(metrics_df["profit_factor"].mean(skipna=True))
    worst_dd = float(metrics_df["max_drawdown_pct"].min())
    n_positive = int((metrics_df["pnl_total"] > 0).sum())

    print(f"Trade totali: {total_trades}")
    print(f"PnL totale (somma 5 periodi, capitale non compounding tra periodi): {total_pnl:+.2f}€")
    print(f"Profit factor medio: {avg_pf:.2f}")
    print(f"Peggior drawdown singolo periodo: {worst_dd:.2f}%")
    print(f"Periodi con PnL positivo: {n_positive}/5")
    print("\nFile: results/official_metrics_5periods.csv (aggregato) + "
          "results/official_trades_<periodo>.csv (dettaglio, per artifact/D1 futuro)")


if __name__ == "__main__":
    main()
