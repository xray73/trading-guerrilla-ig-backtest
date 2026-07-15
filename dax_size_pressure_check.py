"""
dax_size_pressure_check.py — Quantifica la % di trade con size forzata al
minimo negoziabile per DAX e FTSE100, al capitale di riferimento REALE
(2.000€, confermato in chat 15/07/2026) — non i 900€ usati nell'analisi
originale di RCA Addendum 13/07 sez.24, che aveva lasciato la domanda
aperta senza numeri precisi al capitale corretto.

`forced_min_size` è già tracciato per-trade in engine.py (ClosedTrade,
mai esportato in D1 ma presente nel DataFrame in memoria) — nessuna
modifica al motore, solo lettura diretta del DataFrame restituito da
BacktestEngine.run().

Output: results/dax_size_pressure.csv — per strumento e periodo,
n_trade totali, n_forced_min, % forzata al minimo.
"""

from __future__ import annotations
import os
import pandas as pd

import engine as eng
import ema_grid_search as g

CAPITAL0 = 2000.0
INSTRUMENTS_TO_CHECK = ["DAX", "FTSE100"]
ALL_PERIODS = list(g.PERIODS.keys())


def main():
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    rows = []
    for period in ALL_PERIODS:
        data = {}
        for name in INSTRUMENTS_TO_CHECK:
            inst = eng.INSTRUMENTS[name]
            window, period_start = g.slice_period(full_data[name], period)
            sig = eng.generate_signals(window, inst)
            sig = g.trim_warmup(sig, period_start)
            data[name] = sig

        engine_ = eng.BacktestEngine(capital0=CAPITAL0)
        trades_df, metrics_df = engine_.run(data)

        if trades_df.empty:
            for name in INSTRUMENTS_TO_CHECK:
                rows.append({"symbol": name, "period": period, "n_trade": 0,
                             "n_forced_min": 0, "pct_forced_min": None})
            continue

        for name in INSTRUMENTS_TO_CHECK:
            sub = trades_df[trades_df["instrument"] == name]
            n = len(sub)
            n_forced = int(sub["forced_min_size"].sum()) if n else 0
            pct = (n_forced / n * 100) if n else None
            rows.append({"symbol": name, "period": period, "n_trade": n,
                         "n_forced_min": n_forced, "pct_forced_min": pct})
            print(f"  [{period}] {name}: {n_forced}/{n} trade a size forzata "
                  f"({pct:.1f}%)" if n else f"  [{period}] {name}: 0 trade")

    out_df = pd.DataFrame(rows)
    out_df.to_csv("results/dax_size_pressure.csv", index=False)

    print("\n=== RIEPILOGO (capitale reale 2.000€) ===")
    print(out_df.to_string(index=False))
    print("\nCompletato.")


if __name__ == "__main__":
    main()
