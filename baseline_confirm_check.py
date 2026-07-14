"""
baseline_confirm_check.py — Chiude il buco di confronto lasciato dalla grid
search EMA: calcola lo z-score del baseline (20/50 + 100/200) sugli stessi
3 periodi già usati per la fase di conferma del candidato (10/30 + 150/300),
così il confronto candidato-vs-baseline è omogeneo su tutti e 5 i periodi,
non solo su train+test.

Riusa le funzioni già scritte e testate in ema_grid_search.py (stesso
motore, stessa metodologia di baseline random 30 seed) — non reimplementa
nulla, così non c'è rischio di introdurre una discrepanza metodologica tra
questo confronto e quello già fatto.

Richiede DAX_full.csv e FTSE100_full.csv nella working directory (stessi
prodotti da fetch_ohlc_d1.py, già nel repo).

Output: baseline_confirm_3periods.csv con le stesse colonne di
ema_grid_confirm_3periods.csv, per confronto diretto riga per riga.
"""

from __future__ import annotations

import os
import pandas as pd

import ema_grid_search as g

CANDIDATE_FAST = (10, 30)
CANDIDATE_BROAD = (150, 300)


def main():
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    rows = []
    print(f"Calcolo baseline {g.BASELINE_FAST[0]}/{g.BASELINE_FAST[1]} + "
          f"{g.BASELINE_BROAD[0]}/{g.BASELINE_BROAD[1]} sui 3 periodi di conferma "
          f"({g.VALIDATION_RANDOM_SEEDS} seed pieni, stessa rigorosità del candidato)...")

    for period in g.CONFIRM_PERIODS:
        row = g.eval_combo_on_period(g.BASELINE_FAST, g.BASELINE_BROAD, period,
                                      full_data, n_seeds=g.VALIDATION_RANDOM_SEEDS)
        rows.append(row)
        print(f"  [BASELINE {period}] z={row['z_score']:.3f} pnl={row['pnl_total']:.0f} "
              f"trades={row['num_trades']}")

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv("results/baseline_confirm_3periods.csv", index=False)

    # confronto diretto se il file del candidato è presente (portato dal run precedente,
    # opzionale: se non c'è, stampiamo solo il baseline)
    candidate_path = "ema_grid_confirm_3periods.csv"
    if os.path.exists(candidate_path):
        candidate_df = pd.read_csv(candidate_path)
        print("\n=== CONFRONTO DIRETTO (candidato 10/30+150/300 vs baseline 20/50+100/200) ===")
        for period in g.CONFIRM_PERIODS:
            c = candidate_df[candidate_df["period"] == period]
            b = baseline_df[baseline_df["period"] == period]
            if c.empty or b.empty:
                continue
            c_z = c["z_score"].iloc[0]
            b_z = b["z_score"].iloc[0]
            beats = c_z > b_z
            print(f"  {period:12s}  candidato z={c_z:7.3f}   baseline z={b_z:7.3f}   "
                  f"{'CANDIDATO VINCE' if beats else 'baseline tiene/vince'}")
    else:
        print("\n(File ema_grid_confirm_3periods.csv non presente in questa run — "
              "carica anche quello se vuoi il confronto diretto stampato qui; "
              "il CSV con il baseline è comunque salvato in results/.)")

    print("\nCompletato.")


if __name__ == "__main__":
    main()
