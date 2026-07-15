"""
smi_atr_lookback_grid.py — Calibrazione ATR multiplier × breakout
lookback per SMI, stesso principio già usato per GOLD (RCA sez.22.3):
i parametri copiati da DAX (ATR×1.5, lookback 20) sono solo un primo
tentativo grezzo, non necessariamente adatti alla volatilità/scala di
un nuovo strumento.

Metodo: grid search su TRAIN 2023 soltanto (economico), poi verifica
la combo migliore sui restanti 4 periodi. Criterio di promozione
FISSATO PRIMA di vedere risultati (chat 15/07/2026):
  1. Combo scelta su 2023 massimizzando PnL/|drawdown|
  2. Deve dare PnL positivo in almeno 3/4 periodi di verifica
  3. Rapporto aggregato (5 periodi) deve battere la baseline grezza
     attuale (ATR×1.5/lookback20, spread=2.0 corretto) di almeno +10%
  4. Se nessuna combo supera la soglia, si resta sulla baseline grezza
     — nessuna soglia abbassata per far vincere qualcosa

Spread=2.0 (verificato in orario di mercato attivo, 15/07/2026) usato
per OGNI combo del grid — solo ATR/lookback variano.
"""

from __future__ import annotations

import pandas as pd

import engine as eng
import ema_grid_search as g

CAPITAL0 = 2000.0
SYMBOL = "SMI"
SPREAD = 2.0  # verificato, non in grid

ATR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
LOOKBACK_GRID = [15, 20, 30, 40, 50]

TRAIN_PERIOD = "2023"
VERIFY_PERIODS = ["2015-2016", "2020-covid", "2024-2025", "2026-ytd"]
ALL_PERIODS = [TRAIN_PERIOD] + VERIFY_PERIODS

BASELINE_ATR = 1.5
BASELINE_LOOKBACK = 20
PROMOTION_MARGIN = 0.10
MIN_POSITIVE_VERIFY = 3  # su 4


def make_config(atr_mult: float, lookback: int) -> eng.InstrumentConfig:
    return eng.InstrumentConfig(
        name=SYMBOL, tradable=True,
        breakout_lookback=lookback, atr_multiplier=atr_mult,
        risk_pct=0.015, point_value=2.16, spread_fixed=SPREAD,
        min_tradable_size=0.10, margin_pct=0.10,
    )


def run_period(inst: eng.InstrumentConfig, period: str, full_df: pd.DataFrame) -> dict:
    window, period_start = g.slice_period(full_df, period)
    sig = eng.generate_signals(window, inst)
    sig = g.trim_warmup(sig, period_start)

    engine_ = eng.BacktestEngine(capital0=CAPITAL0, instruments={SYMBOL: inst})
    trades_df, metrics_df = engine_.run({SYMBOL: sig})

    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    ratio = pnl / abs(dd) if dd != 0 else 0.0
    return {"period": period, "num_trades": n, "pnl_total": pnl,
            "max_drawdown_pct": dd, "pnl_dd_ratio": ratio}


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_df = g.load_full_ohlc(f"{SYMBOL}_full.csv")

    print(f"=== Grid search {SYMBOL}: {len(ATR_GRID)}x{len(LOOKBACK_GRID)} = "
          f"{len(ATR_GRID)*len(LOOKBACK_GRID)} combo, train={TRAIN_PERIOD} ===\n")

    grid_rows = []
    for atr_mult in ATR_GRID:
        for lookback in LOOKBACK_GRID:
            inst = make_config(atr_mult, lookback)
            r = run_period(inst, TRAIN_PERIOD, full_df)
            r["atr_mult"] = atr_mult
            r["lookback"] = lookback
            grid_rows.append(r)
            print(f"  ATR={atr_mult} LB={lookback}: trades={r['num_trades']} "
                  f"pnl={r['pnl_total']:.1f} dd={r['max_drawdown_pct']*100:.2f}% "
                  f"ratio={r['pnl_dd_ratio']:.1f}")

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv("results/smi_grid_train2023.csv", index=False)

    best = grid_df.loc[grid_df["pnl_dd_ratio"].idxmax()]
    best_atr, best_lb = best["atr_mult"], int(best["lookback"])
    print(f"\n=== Migliore su train 2023: ATR={best_atr} LB={best_lb} "
          f"(ratio={best['pnl_dd_ratio']:.1f}) ===")

    print(f"\n=== Verifica su {VERIFY_PERIODS} ===")
    best_inst = make_config(best_atr, best_lb)
    verify_rows = []
    for period in VERIFY_PERIODS:
        r = run_period(best_inst, period, full_df)
        verify_rows.append(r)
        print(f"  [{period}] trades={r['num_trades']} pnl={r['pnl_total']:.1f} "
              f"dd={r['max_drawdown_pct']*100:.2f}% ratio={r['pnl_dd_ratio']:.1f}")

    verify_df = pd.DataFrame(verify_rows)

    print(f"\n=== Baseline grezza (ATR={BASELINE_ATR} LB={BASELINE_LOOKBACK}) su tutti e 5 i periodi ===")
    baseline_inst = make_config(BASELINE_ATR, BASELINE_LOOKBACK)
    baseline_rows = []
    for period in ALL_PERIODS:
        r = run_period(baseline_inst, period, full_df)
        baseline_rows.append(r)
        print(f"  [{period}] pnl={r['pnl_total']:.1f} dd={r['max_drawdown_pct']*100:.2f}%")
    baseline_df = pd.DataFrame(baseline_rows)

    # combo calibrata sui 5 periodi (train result + verify results)
    train_row = grid_df[(grid_df["atr_mult"] == best_atr) & (grid_df["lookback"] == best_lb)].iloc[0]
    calibrated_all = pd.concat([pd.DataFrame([train_row])[["period", "pnl_total", "max_drawdown_pct"]],
                                 verify_df[["period", "pnl_total", "max_drawdown_pct"]]], ignore_index=True)

    calibrated_sum_pnl = calibrated_all["pnl_total"].sum()
    calibrated_worst_dd = calibrated_all["max_drawdown_pct"].min()
    calibrated_ratio = calibrated_sum_pnl / abs(calibrated_worst_dd) if calibrated_worst_dd != 0 else 0.0

    baseline_sum_pnl = baseline_df["pnl_total"].sum()
    baseline_worst_dd = baseline_df["max_drawdown_pct"].min()
    baseline_ratio = baseline_sum_pnl / abs(baseline_worst_dd) if baseline_worst_dd != 0 else 0.0

    n_positive_verify = int((verify_df["pnl_total"] > 0).sum())
    margin_vs_baseline = (calibrated_ratio / baseline_ratio - 1.0) if baseline_ratio != 0 else float("inf")

    criterio1_ok = True  # per costruzione, la combo è quella scelta massimizzando train
    criterio2_ok = n_positive_verify >= MIN_POSITIVE_VERIFY
    criterio3_ok = margin_vs_baseline >= PROMOTION_MARGIN
    promosso = criterio1_ok and criterio2_ok and criterio3_ok

    print(f"\n{'='*70}")
    print("VERDETTO CALIBRAZIONE")
    print(f"{'='*70}")
    print(f"Combo calibrata: ATR={best_atr} LB={best_lb}")
    print(f"  Somma PnL 5 periodi: {calibrated_sum_pnl:.1f} | worst dd: {calibrated_worst_dd*100:.2f}% | ratio: {calibrated_ratio:.1f}")
    print(f"Baseline grezza (ATR={BASELINE_ATR} LB={BASELINE_LOOKBACK}):")
    print(f"  Somma PnL 5 periodi: {baseline_sum_pnl:.1f} | worst dd: {baseline_worst_dd*100:.2f}% | ratio: {baseline_ratio:.1f}")
    print(f"\nPeriodi verifica positivi: {n_positive_verify}/{len(VERIFY_PERIODS)} "
          f"(criterio2: {'OK' if criterio2_ok else 'FALLITO'}, richiesti >={MIN_POSITIVE_VERIFY})")
    print(f"Margine vs baseline: {margin_vs_baseline*100:+.1f}% "
          f"(criterio3: {'OK' if criterio3_ok else 'FALLITO'}, richiesto >=+{PROMOTION_MARGIN*100:.0f}%)")
    print(f"\n{'PROMOSSA' if promosso else 'NON PROMOSSA — resta la baseline grezza ATR=1.5 LB=20'}")

    summary = pd.DataFrame([{
        "combo": f"ATR={best_atr}_LB={best_lb}",
        "calibrated_sum_pnl": calibrated_sum_pnl, "calibrated_ratio": calibrated_ratio,
        "baseline_sum_pnl": baseline_sum_pnl, "baseline_ratio": baseline_ratio,
        "n_positive_verify": n_positive_verify, "margin_vs_baseline": margin_vs_baseline,
        "PROMOSSA": promosso,
    }])
    summary.to_csv("results/smi_calibration_verdict.csv", index=False)
    verify_df.to_csv("results/smi_calibration_verify_detail.csv", index=False)

    print(f"\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
