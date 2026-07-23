"""
causal_framework.py — Modulo riusabile per test causali (motore vero +
bootstrap a blocchi di giornata), con diagnostica ricca incorporata di
default: breakdown exit_reason, per-strumento, delta/z/CI. Costruito
23/07/2026 dopo aver dovuto aggiungere diagnostica progressivamente,
run dopo run, su test_ftse_dynamic_exit_combined.py — questa volta la
diagnostica c'e' fin dall'inizio, non si aggiunge dopo.

DUE MODALITA' DI CONFRONTO SUPPORTATE (in generale un test causale
puo' differire per motore, per segnali, o per entrambi):
  - stesso motore, segnali diversi (es. controllo positivo: filtro ADX
    presente/assente nel segnale)
  - motore diverso, stessi segnali (es. controllo negativo: motore che
    aggiunge un'uscita casuale scollegata dal segnale)
  - entrambi diversi (caso generale, es. tutti i test di idea 1 oggi)

USO:
  res = bootstrap_compare(
      baseline_engine_factory=lambda cap: BacktestEngineFloatingKillSwitch(capital0=cap),
      baseline_signals=signals_standard,
      variant_engine_factory=lambda cap: BacktestEngineFloatingKillSwitch(capital0=cap),
      variant_signals=signals_variante,
      period_labels=["2020-covid"], periods_dict=PERIODS, capital0=1400.0,
  )
  print_result("Nome test", res)

Bootstrap: blocchi di giornata, N=2000, seed fisso=42 per confrontabilita'
tra run diversi. Nessuna scrittura su D1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def slice_period(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def daily_pnl(trades_df: pd.DataFrame, start: str, end: str) -> pd.Series:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    if trades_df.empty:
        return pd.Series(0.0, index=idx)
    df = trades_df.copy()
    df["exit_day"] = pd.to_datetime(df["exit_time"]).dt.floor("D")
    daily = df.groupby("exit_day")["pnl"].sum()
    return daily.reindex(idx, fill_value=0.0)


def run_period(engine_factory, signals_dict: dict, start: str, end: str, capital0: float):
    """engine_factory: callable(capital0) -> istanza motore con .run(data).
    signals_dict: {instrument: df segnali gia' calcolati}."""
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_dict.items()}
    engine_ = engine_factory(capital0)
    trades_df, _ = engine_.run(sliced)
    return trades_df, engine_


def exit_reason_counts(trades_df: pd.DataFrame) -> dict:
    return trades_df["exit_reason"].value_counts().to_dict() if len(trades_df) else {}


def per_instrument_summary(baseline_trades: pd.DataFrame, variant_trades: pd.DataFrame,
                            instruments=("DAX", "FTSE100")) -> dict:
    out = {}
    for inst_name in instruments:
        b = baseline_trades[baseline_trades["instrument"] == inst_name] if len(baseline_trades) else baseline_trades
        v = variant_trades[variant_trades["instrument"] == inst_name] if len(variant_trades) else variant_trades
        b_pnl = float(b["pnl"].sum()) if len(b) else 0.0
        v_pnl = float(v["pnl"].sum()) if len(v) else 0.0
        out[inst_name] = {
            "delta": v_pnl - b_pnl, "baseline_trades": len(b), "variant_trades": len(v),
            "baseline_exit_counts": exit_reason_counts(b), "variant_exit_counts": exit_reason_counts(v),
        }
    return out


def bootstrap_compare(baseline_engine_factory, baseline_signals: dict,
                       variant_engine_factory, variant_signals: dict,
                       period_labels: list, periods_dict: dict, capital0: float) -> dict:
    all_delta_days = []
    period_summary = []

    for period_name in period_labels:
        start, end = periods_dict[period_name]
        baseline_trades, _ = run_period(baseline_engine_factory, baseline_signals, start, end, capital0)
        variant_trades, variant_engine = run_period(variant_engine_factory, variant_signals, start, end, capital0)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        variant_pnl = float(variant_trades["pnl"].sum()) if len(variant_trades) else 0.0

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_variant = daily_pnl(variant_trades, start, end)
        all_delta_days.append(d_variant - d_baseline)

        period_summary.append({
            "period": period_name, "baseline_pnl": baseline_pnl, "variant_pnl": variant_pnl,
            "delta": variant_pnl - baseline_pnl,
            "baseline_trades": len(baseline_trades), "variant_trades": len(variant_trades),
            "baseline_exit_counts": exit_reason_counts(baseline_trades),
            "variant_exit_counts": exit_reason_counts(variant_trades),
            "per_instrument": per_instrument_summary(baseline_trades, variant_trades),
        })

    combined_deltas = pd.concat(all_delta_days).values
    observed_delta = combined_deltas.sum()
    n_days_total = len(combined_deltas)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_sums = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(combined_deltas, size=n_days_total, replace=True)
        boot_sums[i] = sample.sum()

    std_boot = boot_sums.std()
    z_score = observed_delta / std_boot if std_boot > 0 else float("nan")
    pct_leq_zero = (boot_sums <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_sums, [2.5, 97.5])

    return {
        "observed_delta": observed_delta, "z_score": z_score, "pct_leq_zero": pct_leq_zero,
        "ci_low": ci_low, "ci_high": ci_high, "period_summary": period_summary,
    }


def sanity_check(baseline_engine_factory, baseline_signals: dict,
                  neutral_variant_engine_factory, neutral_variant_signals: dict,
                  check_period: tuple, capital0: float, label: str = "") -> bool:
    """Verifica che variante 'neutra' (force_neutral o config equivalente
    che disattiva ogni meccanismo) riproduca ESATTAMENTE il baseline.
    Ritorna True se ok, altrimenti stampa errore e ritorna False."""
    print(f"=== SANITY CHECK {label} (obbligatorio) ===")
    start, end = check_period
    baseline_trades, _ = run_period(baseline_engine_factory, baseline_signals, start, end, capital0)
    neutral_trades, _ = run_period(neutral_variant_engine_factory, neutral_variant_signals, start, end, capital0)
    n_base, n_neutral = len(baseline_trades), len(neutral_trades)
    pnl_base = float(baseline_trades["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral_trades["pnl"].sum()) if n_neutral else 0.0
    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral: {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")
    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("  *** SANITY CHECK FALLITO ***")
        return False
    print("  OK\n")
    return True


def print_result(label: str, res: dict):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    for s in res["period_summary"]:
        print(f"  {s['period']:<12} trade base={s['baseline_trades']:>4} var={s['variant_trades']:>4}  "
              f"PnL base={s['baseline_pnl']:>10.2f}  PnL var={s['variant_pnl']:>10.2f}  "
              f"delta={s['delta']:>+9.2f}")
        print(f"      exit_reason base: {s['baseline_exit_counts']}")
        print(f"      exit_reason var:  {s['variant_exit_counts']}")
        for inst_name, pi in s["per_instrument"].items():
            print(f"      {inst_name:<8} delta={pi['delta']:>+9.2f}  "
                  f"trade base={pi['baseline_trades']:>4} var={pi['variant_trades']:>4}")
    print(f"\n  Delta osservato: {res['observed_delta']:+.2f} EUR")
    print(f"  Z-score: {res['z_score']:.3f}")
    print(f"  %% iterazioni con delta<=0: {res['pct_leq_zero']:.1f}%%")
    print(f"  95%% CI bootstrap: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")
