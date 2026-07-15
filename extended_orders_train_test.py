"""
extended_orders_train_test.py — Rifà la valutazione degli slot extra
(4°/5° trade) con la disciplina corretta: train (4 periodi uniti) /
test (2026-ytd, mai visto) + stima del rumore via bootstrap — stesso
protocollo già validato per il regime ATR.

CORREZIONE METODOLOGICA rispetto al primo giro (extended_orders_impact.py,
15/07/2026): quel test misurava l'impatto sugli STESSI 5 periodi usati
per progettare il meccanismo — nessun dato mai visto a verificarlo.
Qui si separa train e test come per ogni altro test del progetto.

Non usa una soglia percentuale fissa importata da un altro contesto
(il +10% dei test di uscita, mai validato per questo caso) — la
decisione si basa sul confronto tra il margine osservato sul test e
la stima del rumore di ricampionamento dello stesso periodo. Se il
margine osservato è chiaramente fuori dalla banda di rumore, è un
segnale reale; se ci sta comodamente dentro, è indistinguibile dalla
fortuna, qualunque soglia percentuale si scelga.

Parametri del meccanismo (invariati, già fissati in chat): max 5
ordini/giorno (invece di 3), slot 4-5 con rischio modulato,
extra_slot_pct=1.0, tetto al rischio standard dello strumento.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_extended_orders import BacktestEngineExtendedOrders

CAPITAL0 = 2000.0
TRAIN_PERIODS = ["2015-2016", "2020-covid", "2023", "2024-2025"]
TEST_PERIOD = "2026-ytd"
EXTRA_SLOT_PCT = 1.0
N_BOOTSTRAP = 300


def run_period(period: str, full_data: dict, p) -> dict:
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig

    kwargs = {"capital0": CAPITAL0, "p": p}
    if p.max_new_orders_per_day > 3:
        kwargs["extra_slot_pct"] = EXTRA_SLOT_PCT
    engine_ = BacktestEngineExtendedOrders(**kwargs)
    trades_df, metrics_df = engine_.run(data)

    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return {"period": period, "num_trades": n, "pnl_total": pnl,
            "max_drawdown_pct": dd, "trades_df": trades_df}


def aggregate(results: list[dict]) -> tuple[float, float]:
    sum_pnl = sum(r["pnl_total"] for r in results)
    worst_dd = min(r["max_drawdown_pct"] for r in results)
    return sum_pnl, worst_dd


def bootstrap_noise(trades_df: pd.DataFrame, capital0: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """Stesso principio del bootstrap già usato per l'ATR regime:
    resampling a blocchi di giornata (Monte Carlo del progetto)."""
    if trades_df.empty:
        return {"ratio_std": np.nan, "ratio_ci_low": np.nan, "ratio_ci_high": np.nan, "ratio_median": np.nan}

    trades_df = trades_df.copy()
    trades_df["day"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    daily_pnl = trades_df.groupby("day")["pnl"].sum()
    days = daily_pnl.index.tolist()
    n_days = len(days)
    if n_days < 5:
        return {"ratio_std": np.nan, "ratio_ci_low": np.nan, "ratio_ci_high": np.nan, "ratio_median": np.nan}

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

    p_baseline = eng.PARAMS  # max_new_orders_per_day=3
    p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=5)

    print("=== TRAIN (4 periodi uniti, solo informativo — non decide nulla) ===")
    baseline_train = [run_period(p, full_data, p_baseline) for p in TRAIN_PERIODS]
    extended_train = [run_period(p, full_data, p_extended) for p in TRAIN_PERIODS]

    baseline_train_pnl, baseline_train_dd = aggregate(baseline_train)
    extended_train_pnl, extended_train_dd = aggregate(extended_train)
    baseline_train_ratio = baseline_train_pnl / abs(baseline_train_dd) if baseline_train_dd != 0 else 0.0
    extended_train_ratio = extended_train_pnl / abs(extended_train_dd) if extended_train_dd != 0 else 0.0
    train_margin = (extended_train_ratio / baseline_train_ratio - 1.0) if baseline_train_ratio != 0 else float("inf")

    print(f"  Baseline: pnl={baseline_train_pnl:.1f} worst_dd={baseline_train_dd*100:.2f}% ratio={baseline_train_ratio:.1f}")
    print(f"  Esteso  : pnl={extended_train_pnl:.1f} worst_dd={extended_train_dd*100:.2f}% ratio={extended_train_ratio:.1f}")
    print(f"  Margine train: {train_margin*100:+.1f}% (informativo, non decide nulla)")

    print(f"\n=== TEST ({TEST_PERIOD}, mai visto — questo decide) ===")
    baseline_test = run_period(TEST_PERIOD, full_data, p_baseline)
    extended_test = run_period(TEST_PERIOD, full_data, p_extended)

    baseline_test_ratio = (baseline_test["pnl_total"] / abs(baseline_test["max_drawdown_pct"])
                            if baseline_test["max_drawdown_pct"] != 0 else 0.0)
    extended_test_ratio = (extended_test["pnl_total"] / abs(extended_test["max_drawdown_pct"])
                            if extended_test["max_drawdown_pct"] != 0 else 0.0)
    test_margin = (extended_test_ratio / baseline_test_ratio - 1.0) if baseline_test_ratio != 0 else float("inf")

    print(f"  Baseline: trade={baseline_test['num_trades']} pnl={baseline_test['pnl_total']:.1f} "
          f"dd={baseline_test['max_drawdown_pct']*100:.2f}% ratio={baseline_test_ratio:.1f}")
    print(f"  Esteso  : trade={extended_test['num_trades']} pnl={extended_test['pnl_total']:.1f} "
          f"dd={extended_test['max_drawdown_pct']*100:.2f}% ratio={extended_test_ratio:.1f}")
    print(f"  Margine test osservato: {test_margin*100:+.1f}%")

    print(f"\n=== STIMA RUMORE (bootstrap {N_BOOTSTRAP} resample a blocchi di giornata sul TEST) ===")
    noise_baseline = bootstrap_noise(baseline_test["trades_df"], CAPITAL0)
    noise_extended = bootstrap_noise(extended_test["trades_df"], CAPITAL0)

    print(f"  Baseline: ratio mediano bootstrap={noise_baseline['ratio_median']:.1f}, "
          f"IC95%=[{noise_baseline['ratio_ci_low']:.1f}, {noise_baseline['ratio_ci_high']:.1f}], "
          f"std={noise_baseline['ratio_std']:.1f}")
    print(f"  Esteso  : ratio mediano bootstrap={noise_extended['ratio_median']:.1f}, "
          f"IC95%=[{noise_extended['ratio_ci_low']:.1f}, {noise_extended['ratio_ci_high']:.1f}], "
          f"std={noise_extended['ratio_std']:.1f}")

    # il margine osservato è "rumore" o "reale"? confronto contro la
    # deviazione standard del rumore stimato (non contro una soglia %
    # arbitraria importata da un altro contesto)
    valid_stds = [s for s in [noise_baseline["ratio_std"], noise_extended["ratio_std"]] if pd.notna(s)]
    avg_noise_std = np.mean(valid_stds) if valid_stds else np.nan
    observed_diff = extended_test_ratio - baseline_test_ratio
    signal_to_noise = observed_diff / avg_noise_std if avg_noise_std and not np.isnan(avg_noise_std) else np.nan

    overlap = None
    if pd.notna(noise_baseline["ratio_ci_high"]) and pd.notna(noise_extended["ratio_ci_low"]):
        overlap = noise_extended["ratio_ci_low"] < noise_baseline["ratio_ci_high"]

    print(f"\n  Differenza osservata sul ratio: {observed_diff:+.1f}")
    print(f"  Deviazione standard media del rumore: {avg_noise_std:.1f}")
    print(f"  Rapporto segnale/rumore: {signal_to_noise:.2f} "
          f"({'oltre 1 deviazione standard, segnale plausibilmente reale' if abs(signal_to_noise) > 1 else 'sotto 1 deviazione standard, indistinguibile dal rumore'})")
    print(f"  Sovrapposizione intervalli 95%: {'SÌ' if overlap else 'NO'}")

    summary = pd.DataFrame([{
        "train_margin_pct": train_margin, "test_margin_pct": test_margin,
        "test_baseline_ratio": baseline_test_ratio, "test_extended_ratio": extended_test_ratio,
        "noise_baseline_std": noise_baseline["ratio_std"], "noise_extended_std": noise_extended["ratio_std"],
        "signal_to_noise_ratio": signal_to_noise, "noise_ci_overlap": overlap,
    }])
    summary.to_csv("results/extended_orders_train_test_verdict.csv", index=False)

    print(f"\nCompletato. File: results/extended_orders_train_test_verdict.csv")
    print("Nessuna soglia percentuale fissa applicata — la decisione va presa "
          "guardando il rapporto segnale/rumore sopra, discutendone insieme.")


if __name__ == "__main__":
    main()
