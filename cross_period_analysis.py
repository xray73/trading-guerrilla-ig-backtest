"""
cross_period_analysis.py — Separa le statistiche dei trade "slot
standard" (1-3) da quelli "slot extra" (4-5) per ciascuno dei 5
periodi, per capire se il peggioramento di 2024-2025 (visto in
extended_orders_impact.csv) è un problema di VOLUME (troppi slot
extra aperti, diluizione) o di QUALITÀ (win rate sistematicamente
peggiore in quel periodo specifico) — due letture diverse con
implicazioni diverse.

Aggiunge anche ATR medio del periodo (proxy di volatilità) come primo
elemento di contesto quantitativo, calcolato direttamente dai dati
OHLC già disponibili (nessuna nuova fonte dati).
"""

from __future__ import annotations

import dataclasses
import pandas as pd
import numpy as np

import engine as eng
import ema_grid_search as g
from engine_extended_orders import BacktestEngineExtendedOrders

CAPITAL0 = 2000.0
EXTRA_SLOT_PCT = 1.0


def run_and_tag(period: str, full_data: dict) -> tuple[pd.DataFrame, dict]:
    data = {}
    atr_values = []
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig
        atr_values.append(sig["atr"].dropna())

    atr_all = pd.concat(atr_values)
    atr_mean = float(atr_all.mean())
    atr_median = float(atr_all.median())

    p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=5)
    engine_ = BacktestEngineExtendedOrders(capital0=CAPITAL0, p=p_extended, extra_slot_pct=EXTRA_SLOT_PCT)
    trades_df, metrics_df = engine_.run(data)

    extra_keys = set(engine_.extra_slot_log)
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["is_extra_slot"] = trades_df.apply(
            lambda r: (r["instrument"], r["entry_time"]) in extra_keys, axis=1)
    else:
        trades_df["is_extra_slot"] = pd.Series(dtype=bool)

    context = {
        "period": period, "atr_mean": atr_mean, "atr_median": atr_median,
        "n_extra_opened": engine_.n_extra_slot_opened,
        "n_extra_skipped_pnl": engine_.n_extra_slot_skipped_pnl,
        "n_extra_skipped_minsize": engine_.n_extra_slot_skipped_min_size,
    }
    return trades_df, context


def summarize(trades_df: pd.DataFrame, period: str, context: dict) -> dict:
    standard = trades_df[~trades_df["is_extra_slot"]]
    extra = trades_df[trades_df["is_extra_slot"]]

    def stats(sub, label):
        if sub.empty:
            return {f"{label}_n": 0, f"{label}_winrate": np.nan, f"{label}_avg_pnl": np.nan,
                    f"{label}_sum_pnl": 0.0}
        winrate = (sub["pnl"] > 0).mean()
        return {
            f"{label}_n": len(sub), f"{label}_winrate": winrate,
            f"{label}_avg_pnl": sub["pnl"].mean(), f"{label}_sum_pnl": sub["pnl"].sum(),
        }

    row = {"period": period, **context}
    row.update(stats(standard, "standard"))
    row.update(stats(extra, "extra"))
    return row


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    rows = []
    for period in g.PERIODS:
        trades_df, context = run_and_tag(period, full_data)
        row = summarize(trades_df, period, context)
        rows.append(row)
        print(f"[{period}] ATR medio={context['atr_mean']:.1f} | "
              f"standard: n={row['standard_n']} winrate={row['standard_winrate']*100 if pd.notna(row['standard_winrate']) else float('nan'):.1f}% "
              f"avg_pnl={row['standard_avg_pnl']:.1f} | "
              f"extra: n={row['extra_n']} winrate={row['extra_winrate']*100 if pd.notna(row['extra_winrate']) else float('nan'):.1f}% "
              f"avg_pnl={row['extra_avg_pnl']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/cross_period_analysis.csv", index=False)
    print("\nCompletato. File: results/cross_period_analysis.csv")


if __name__ == "__main__":
    main()
