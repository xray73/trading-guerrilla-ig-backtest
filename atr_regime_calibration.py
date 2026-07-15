"""
atr_regime_calibration.py — Calibra i moltiplicatori di rischio per
fascia di volatilità (basso/medio/alto percentile ATR rolling),
applicati ai 3 slot BASE del segnale.

TRAIN: 2015-2016 + 2020-covid + 2023 + 2024-2025 (uniti, aggregando
PnL e usando il worst-case drawdown tra i 4 come denominatore) — copre
tutti i regimi di volatilità disponibili tranne l'ultimo.
TEST: 2026-ytd (mai visto durante la calibrazione) — rispetta la
disciplina temporale (nessun dato futuro usato per calibrare un
periodo passato).

Criterio di promozione (fissato in chat 15/07/2026): la combo
migliore su train deve battere la baseline (nessuna modulazione,
moltiplicatori tutti 1.0) di almeno +10% sul rapporto PnL/drawdown,
verificato sul periodo di TEST.

STIMA DEL RUMORE (richiesta esplicitamente, informativa — NON entra
nel criterio di successo/fallimento): bootstrap a blocchi di giornata
sul periodo di test, stesso principio del Monte Carlo già in uso nel
progetto (resampling a blocchi, non trade-by-trade). Riporta quanto
il margine osservato è largo rispetto alla variabilità naturale di
ricampionamento dello stesso periodo — solo contesto per giudicare
quanto fidarsi del numero, la decisione promuovi/non promuovi resta
sul margine osservato reale, non sulla stima di rumore.
"""

from __future__ import annotations

import dataclasses
import itertools
import numpy as np
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_atr_regime import BacktestEngineATRRegime, compute_atr_regime

CAPITAL0 = 2000.0
TRAIN_PERIODS = ["2015-2016", "2020-covid", "2023", "2024-2025"]
TEST_PERIOD = "2026-ytd"
WINDOW_CANDIDATES = [20, 40, 90]
MULT_CANDIDATES = [0.5, 1.0, 1.5]
PROMOTION_MARGIN = 0.10
N_BOOTSTRAP = 300


def prepare_period_data(period: str, full_data: dict, window_days: int) -> dict:
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        sig = compute_atr_regime(sig, window_days=window_days)
        data[name] = sig
    return data


def run_config(period: str, full_data: dict, window_days: int, multipliers: dict) -> dict:
    data = prepare_period_data(period, full_data, window_days)
    engine_ = BacktestEngineATRRegime(capital0=CAPITAL0, tier_multipliers=multipliers)
    trades_df, metrics_df = engine_.run(data)

    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return {"period": period, "num_trades": n, "pnl_total": pnl,
            "max_drawdown_pct": dd, "trades_df": trades_df}


def aggregate_train(results: list[dict]) -> tuple[float, float]:
    sum_pnl = sum(r["pnl_total"] for r in results)
    worst_dd = min(r["max_drawdown_pct"] for r in results)
    return sum_pnl, worst_dd


def bootstrap_noise(trades_df: pd.DataFrame, capital0: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """Resampling a blocchi di giornata (stesso principio del Monte Carlo
    del progetto). Ritorna distribuzione di PnL/drawdown ricampionati."""
    if trades_df.empty:
        return {"ratio_std": np.nan, "ratio_ci_low": np.nan, "ratio_ci_high": np.nan}

    trades_df = trades_df.copy()
    trades_df["day"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    daily_pnl = trades_df.groupby("day")["pnl"].sum()
    days = daily_pnl.index.tolist()
    n_days = len(days)
    if n_days < 5:
        return {"ratio_std": np.nan, "ratio_ci_low": np.nan, "ratio_ci_high": np.nan}

    rng = np.random.default_rng(42)
    ratios = []
    for _ in range(n_boot):
        sampled_days = rng.choice(days, size=n_days, replace=True)
        sampled_pnls = daily_pnl.loc[sampled_days].values
        equity = capital0 + np.cumsum(sampled_pnls)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        total_pnl = sampled_pnls.sum()
        if max_dd != 0:
            ratios.append(total_pnl / abs(max_dd))

    ratios = np.array(ratios)
    return {
        "ratio_std": float(np.std(ratios)),
        "ratio_ci_low": float(np.percentile(ratios, 2.5)),
        "ratio_ci_high": float(np.percentile(ratios, 97.5)),
        "ratio_median": float(np.median(ratios)),
    }


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    baseline_mult = {"low": 1.0, "medium": 1.0, "high": 1.0}

    print("=== BASELINE (nessuna modulazione) su TRAIN ===")
    baseline_train_results = [run_config(p, full_data, 40, baseline_mult) for p in TRAIN_PERIODS]
    baseline_train_pnl, baseline_train_dd = aggregate_train(baseline_train_results)
    baseline_train_ratio = baseline_train_pnl / abs(baseline_train_dd) if baseline_train_dd != 0 else 0.0
    print(f"  Train: pnl={baseline_train_pnl:.1f} worst_dd={baseline_train_dd*100:.2f}% "
          f"ratio={baseline_train_ratio:.1f}")

    print("\n=== GRID SEARCH su TRAIN (4 periodi uniti) ===")
    grid_rows = []
    all_mult_combos = list(itertools.product(MULT_CANDIDATES, repeat=3))  # (low, medium, high)
    # ESCLUDE le combo uniformi (low==medium==high): quelle equivalgono a
    # "più rischio ovunque", non a una vera differenziazione di regime —
    # scoperto nel primo giro (15/07/2026): la combo uniforme 1.5x/1.5x/1.5x
    # vinceva il grid search senza rappresentare l'ipotesi che si vuole
    # testare. Qui isoliamo SOLO le combo genuinamente differenziate.
    mult_combos = [c for c in all_mult_combos if not (c[0] == c[1] == c[2])]
    total_combos = len(WINDOW_CANDIDATES) * len(mult_combos)
    print(f"Totale combinazioni (SOLO differenziate, uniformi escluse): {total_combos} "
          f"({len(all_mult_combos) - len(mult_combos)} uniformi escluse su {len(all_mult_combos)} totali)")

    for window_days in WINDOW_CANDIDATES:
        for low_m, med_m, high_m in mult_combos:
            multipliers = {"low": low_m, "medium": med_m, "high": high_m}
            train_results = [run_config(p, full_data, window_days, multipliers) for p in TRAIN_PERIODS]
            sum_pnl, worst_dd = aggregate_train(train_results)
            ratio = sum_pnl / abs(worst_dd) if worst_dd != 0 else 0.0
            grid_rows.append({
                "window_days": window_days, "low_mult": low_m, "medium_mult": med_m, "high_mult": high_m,
                "train_sum_pnl": sum_pnl, "train_worst_dd": worst_dd, "train_ratio": ratio,
            })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv("results/atr_regime_grid_train_differentiated.csv", index=False)

    best = grid_df.loc[grid_df["train_ratio"].idxmax()]
    best_window = int(best["window_days"])
    best_mult = {"low": best["low_mult"], "medium": best["medium_mult"], "high": best["high_mult"]}
    print(f"\nMigliore su train: finestra={best_window}gg, moltiplicatori={best_mult} "
          f"(ratio train={best['train_ratio']:.1f})")

    print(f"\n=== VERIFICA su TEST ({TEST_PERIOD}, mai visto) ===")
    baseline_test = run_config(TEST_PERIOD, full_data, 40, baseline_mult)  # finestra irrilevante per baseline (mult=1 ovunque)
    best_test = run_config(TEST_PERIOD, full_data, best_window, best_mult)

    baseline_test_ratio = (baseline_test["pnl_total"] / abs(baseline_test["max_drawdown_pct"])
                            if baseline_test["max_drawdown_pct"] != 0 else 0.0)
    best_test_ratio = (best_test["pnl_total"] / abs(best_test["max_drawdown_pct"])
                        if best_test["max_drawdown_pct"] != 0 else 0.0)

    margin = (best_test_ratio / baseline_test_ratio - 1.0) if baseline_test_ratio != 0 else float("inf")
    insufficient_data = baseline_test["num_trades"] == 0 or best_test["num_trades"] == 0
    promoted = (margin >= PROMOTION_MARGIN) and not insufficient_data
    if insufficient_data:
        print(f"  ATTENZIONE: baseline o combo con 0 trade nel periodo di test — "
              f"margine non significativo, NON promossa a prescindere dal numero.")

    print(f"  Baseline test: trade={baseline_test['num_trades']} pnl={baseline_test['pnl_total']:.1f} "
          f"dd={baseline_test['max_drawdown_pct']*100:.2f}% ratio={baseline_test_ratio:.1f}")
    print(f"  Combo migliore test: trade={best_test['num_trades']} pnl={best_test['pnl_total']:.1f} "
          f"dd={best_test['max_drawdown_pct']*100:.2f}% ratio={best_test_ratio:.1f}")
    print(f"  Margine: {margin*100:+.1f}% (soglia richiesta: +{PROMOTION_MARGIN*100:.0f}%)")
    print(f"  {'PROMOSSA' if promoted else 'NON promossa'}")

    print(f"\n=== STIMA RUMORE (bootstrap {N_BOOTSTRAP} resample a blocchi di giornata, "
          f"SOLO INFORMATIVO) ===")
    noise_baseline = bootstrap_noise(baseline_test["trades_df"], CAPITAL0)
    noise_best = bootstrap_noise(best_test["trades_df"], CAPITAL0)
    print(f"  Baseline: ratio mediano bootstrap={noise_baseline.get('ratio_median', float('nan')):.1f}, "
          f"IC95%=[{noise_baseline.get('ratio_ci_low', float('nan')):.1f}, "
          f"{noise_baseline.get('ratio_ci_high', float('nan')):.1f}], "
          f"std={noise_baseline.get('ratio_std', float('nan')):.1f}")
    print(f"  Combo migliore: ratio mediano bootstrap={noise_best.get('ratio_median', float('nan')):.1f}, "
          f"IC95%=[{noise_best.get('ratio_ci_low', float('nan')):.1f}, "
          f"{noise_best.get('ratio_ci_high', float('nan')):.1f}], "
          f"std={noise_best.get('ratio_std', float('nan')):.1f}")

    overlap = None
    if pd.notna(noise_baseline.get("ratio_ci_high")) and pd.notna(noise_best.get("ratio_ci_low")):
        overlap = noise_best["ratio_ci_low"] < noise_baseline["ratio_ci_high"]
        print(f"  Sovrapposizione intervalli 95%: {'SÌ (il margine potrebbe essere rumore)' if overlap else 'NO (margine oltre il rumore tipico)'}")

    summary = pd.DataFrame([{
        "best_window_days": best_window, "best_low_mult": best_mult["low"],
        "best_medium_mult": best_mult["medium"], "best_high_mult": best_mult["high"],
        "test_baseline_ratio": baseline_test_ratio, "test_best_ratio": best_test_ratio,
        "margin_pct": margin, "PROMOSSA": promoted,
        "noise_baseline_std": noise_baseline.get("ratio_std"),
        "noise_best_std": noise_best.get("ratio_std"),
        "noise_ci_overlap": overlap,
    }])
    summary.to_csv("results/atr_regime_verdict_differentiated.csv", index=False)

    print(f"\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
