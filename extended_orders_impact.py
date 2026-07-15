"""
extended_orders_impact.py — Misura l'impatto reale degli slot extra
(4°, 5°) sui 5 periodi standard, DAX+FTSE100, capitale 2.000€.
Confronta baseline (max 3 ordini/giorno, floating kill switch) contro
esteso (max 5 ordini/giorno, stesso kill switch, slot 4-5 modulati dal
PnL di giornata, extra_slot_pct=1.0 — rischio massimo slot extra =
100% del PnL netto già realizzato in giornata, con tetto al rischio
standard dello strumento).
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_extended_orders import BacktestEngineExtendedOrders

CAPITAL0 = 2000.0
EXTRA_SLOT_PCT = 1.0
MAX_ORDERS_EXTENDED = 5


def run_period(period: str, full_data: dict, p, extra_slot_pct=None) -> dict:
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig

    kwargs = {"capital0": CAPITAL0, "p": p}
    if extra_slot_pct is not None:
        kwargs["extra_slot_pct"] = extra_slot_pct
    engine_ = BacktestEngineExtendedOrders(**kwargs)
    trades_df, metrics_df = engine_.run(data)

    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return {
        "period": period, "num_trades": n, "pnl_total": pnl, "max_drawdown_pct": dd,
        "n_extra_opened": engine_.n_extra_slot_opened,
        "n_extra_skipped_pnl": engine_.n_extra_slot_skipped_pnl,
        "n_extra_skipped_minsize": engine_.n_extra_slot_skipped_min_size,
    }


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    p_base = eng.PARAMS  # max_new_orders_per_day=3
    p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=MAX_ORDERS_EXTENDED)

    rows = []
    for period in g.PERIODS:
        baseline = run_period(period, full_data, p_base)
        baseline["versione"] = "baseline_3ordini"
        rows.append(baseline)

        extended = run_period(period, full_data, p_extended, extra_slot_pct=EXTRA_SLOT_PCT)
        extended["versione"] = "esteso_5ordini"
        rows.append(extended)

        delta_pnl = extended["pnl_total"] - baseline["pnl_total"]
        delta_dd = extended["max_drawdown_pct"] - baseline["max_drawdown_pct"]
        print(f"[{period}] baseline: {baseline['num_trades']} trade, pnl={baseline['pnl_total']:.1f}, "
              f"dd={baseline['max_drawdown_pct']*100:.2f}%")
        print(f"           esteso  : {extended['num_trades']} trade, pnl={extended['pnl_total']:.1f}, "
              f"dd={extended['max_drawdown_pct']*100:.2f}% "
              f"(extra aperti: {extended['n_extra_opened']}, "
              f"saltati pnl<=0: {extended['n_extra_skipped_pnl']}, "
              f"saltati size min: {extended['n_extra_skipped_minsize']})")
        print(f"           differenza: {delta_pnl:+.1f}€ PnL, "
              f"drawdown {'peggiorato' if delta_dd < 0 else 'migliorato/invariato'} "
              f"di {abs(delta_dd)*100:.2f} punti\n")

    df = pd.DataFrame(rows)
    df.to_csv("results/extended_orders_impact.csv", index=False)

    base_df = df[df["versione"] == "baseline_3ordini"]
    ext_df = df[df["versione"] == "esteso_5ordini"]
    print("=" * 70)
    print("RIEPILOGO AGGREGATO (5 periodi)")
    print("=" * 70)
    print(f"Baseline (3 ordini/giorno): {base_df['num_trades'].sum()} trade totali, "
          f"PnL {base_df['pnl_total'].sum():.1f}€, peggior dd {base_df['max_drawdown_pct'].min()*100:.2f}%")
    print(f"Esteso   (5 ordini/giorno): {ext_df['num_trades'].sum()} trade totali, "
          f"PnL {ext_df['pnl_total'].sum():.1f}€, peggior dd {ext_df['max_drawdown_pct'].min()*100:.2f}%")
    print(f"\nGuadagno PnL aggregato: {ext_df['pnl_total'].sum() - base_df['pnl_total'].sum():+.1f}€ "
          f"({(ext_df['pnl_total'].sum()/base_df['pnl_total'].sum()-1)*100:+.1f}%)")
    print(f"Variazione peggior drawdown: "
          f"{(ext_df['max_drawdown_pct'].min() - base_df['max_drawdown_pct'].min())*100:+.2f} punti")
    print(f"Slot extra totali aperti: {ext_df['n_extra_opened'].sum()}")

    print("\nCompletato. File: results/extended_orders_impact.csv")


if __name__ == "__main__":
    main()
