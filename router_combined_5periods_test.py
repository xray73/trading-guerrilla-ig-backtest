"""
router_combined_5periods_test.py — Stesso confronto separati vs
combinati (17-18/07/2026) ma sui 5 periodi ufficiali, non solo gli
ultimi 180 giorni — per capire se il vantaggio di rendimento visto sul
campione recente (compounding incrociato) e il costo in drawdown
(diversificazione persa) si confermano su periodi/regimi diversi, o
erano un artefatto di quella singola finestra.

Ogni periodo è indipendente (capitale riparte da CAPITAL0 in entrambi
gli scenari), coerente con la metodologia walk-forward del progetto.

Mean-reversion: variante RSI (PF più alto nel test di fattibilità).
"""

from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from combined_router_signals import generate_combined_signals
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
MR_MODE = "rsi"
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]


def fetch_bars(symbol_const, start, end) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_raw(period_start: str, period_end: str) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    raw = {}
    for name, const in SYMBOLS.items():
        raw[name] = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
    return raw, p_start


def slice_from(signals_df: pd.DataFrame, p_start: pd.Timestamp) -> pd.DataFrame:
    return signals_df[signals_df["timestamp"] >= p_start].reset_index(drop=True)


def compute_drawdown(trades_df: pd.DataFrame, capital0: float) -> tuple[float, float]:
    if trades_df.empty:
        return 0.0, 0.0
    trades_sorted = trades_df.sort_values("exit_time")
    equity = capital0 + trades_sorted["pnl"].cumsum()
    running_max = equity.cummax()
    dd_eur = equity - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "pnl_total": 0.0,
                "max_dd_pct": np.nan, "max_dd_eur": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    dd_pct, dd_eur = compute_drawdown(trades_df, CAPITAL0)
    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(),
            "max_dd_pct": dd_pct, "max_dd_eur": dd_eur}


def compute_combined_wealth_drawdown(trades_a: pd.DataFrame, trades_b: pd.DataFrame, capital0_each: float) -> tuple[float, float]:
    events = []
    for _, t in trades_a.iterrows():
        events.append((pd.Timestamp(t["exit_time"]), "a", t["pnl"]))
    for _, t in trades_b.iterrows():
        events.append((pd.Timestamp(t["exit_time"]), "b", t["pnl"]))
    events.sort(key=lambda e: e[0])

    cap_a, cap_b = capital0_each, capital0_each
    totals = []
    for _, which, pnl in events:
        if which == "a":
            cap_a += pnl
        else:
            cap_b += pnl
        totals.append(cap_a + cap_b)

    if not totals:
        return 0.0, 0.0
    totals = pd.Series(totals)
    running_max = totals.cummax()
    dd_eur = totals - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def main():
    print(f"=== Separati vs Combinati (router) sui 5 periodi ufficiali — mean-reversion mode={MR_MODE} ===\n")

    all_rows = []
    for label, p_start_str, p_end_str in PERIODS:
        print(f"\n{'='*70}\nPeriodo {label}\n{'='*70}")
        raw_data, p_start = get_period_raw(p_start_str, p_end_str)

        v6_signal_data = {name: slice_from(eng.generate_signals(raw_data[name], eng.INSTRUMENTS[name]), p_start)
                           for name in SYMBOLS}
        mr_signal_data = {name: slice_from(generate_mean_reversion_signals(raw_data[name], eng.INSTRUMENTS[name], mode=MR_MODE), p_start)
                           for name in SYMBOLS}

        engine_v6 = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_v6, _ = engine_v6.run(v6_signal_data)

        engine_mr = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_mr, _ = engine_mr.run(mr_signal_data)

        m_v6 = metrics_summary(trades_v6)
        m_mr = metrics_summary(trades_mr)
        separated_pnl = m_v6["pnl_total"] + m_mr["pnl_total"]
        separated_capital = CAPITAL0 * 2
        separated_dd_pct, separated_dd_eur = compute_combined_wealth_drawdown(trades_v6, trades_mr, CAPITAL0)

        combined_signal_data = {name: slice_from(generate_combined_signals(raw_data[name], eng.INSTRUMENTS[name], mr_mode=MR_MODE), p_start)
                                 for name in SYMBOLS}
        engine_combined = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_combined, _ = engine_combined.run(combined_signal_data)
        m_combined = metrics_summary(trades_combined)

        sep_ret_pct = 100 * separated_pnl / separated_capital
        comb_ret_pct = 100 * m_combined["pnl_total"] / CAPITAL0

        print(f"  SEPARATI:  n_tot={m_v6['n_trades']+m_mr['n_trades']} rendimento={sep_ret_pct:+.1f}% "
              f"MaxDD={separated_dd_pct:.1f}%")
        print(f"  COMBINATI: n={m_combined['n_trades']} rendimento={comb_ret_pct:+.1f}% "
              f"MaxDD={m_combined['max_dd_pct']:.1f}%")

        all_rows.append({
            "periodo": label,
            "sep_n_trades": m_v6["n_trades"] + m_mr["n_trades"],
            "sep_rendimento_pct": sep_ret_pct, "sep_max_dd_pct": separated_dd_pct,
            "sep_v6_pnl": m_v6["pnl_total"], "sep_mr_pnl": m_mr["pnl_total"],
            "comb_n_trades": m_combined["n_trades"],
            "comb_rendimento_pct": comb_ret_pct, "comb_max_dd_pct": m_combined["max_dd_pct"],
        })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("router_combined_5periods_summary.csv", index=False)

    print(f"\n{'='*70}\nRIEPILOGO — medie sui 5 periodi\n{'='*70}")
    print(f"Rendimento medio: separati {summary_df['sep_rendimento_pct'].mean():+.1f}%  "
          f"combinati {summary_df['comb_rendimento_pct'].mean():+.1f}%")
    print(f"MaxDD medio: separati {summary_df['sep_max_dd_pct'].mean():.1f}%  "
          f"combinati {summary_df['comb_max_dd_pct'].mean():.1f}%")
    print(f"Periodi in cui i combinati rendono di più: "
          f"{(summary_df['comb_rendimento_pct'] > summary_df['sep_rendimento_pct']).sum()}/5")
    print(f"Periodi in cui i combinati hanno drawdown peggiore (piu negativo): "
          f"{(summary_df['comb_max_dd_pct'] < summary_df['sep_max_dd_pct']).sum()}/5")

    print("\nFile: router_combined_5periods_summary.csv")


if __name__ == "__main__":
    main()
