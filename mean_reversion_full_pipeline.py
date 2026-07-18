"""
mean_reversion_full_pipeline.py — Pipeline unica (18/07/2026) per non
dover rifare più giri separati:

  FASE 1 — Fattibilità corretta: Bollinger vs RSI, con
    BacktestEngineMeanReversion (salta invece di forzare la size
    minima) — i numeri di fattibilità precedenti erano inquinati dalla
    size forzata al 100% su DAX, quindi vanno riconfermati da zero.
    Seleziona automaticamente la variante con PF più alto per la FASE 2.

  FASE 2 — Test di split capitale REALE (2.000 EUR, non 4.000): stesse
    5 proporzioni di prima (90/10..50/50), sui 5 periodi ufficiali,
    con la variante mean-reversion selezionata in FASE 1 e il motore
    corretto. Include anche V6 DA SOLA con 2.000 EUR interi come
    riferimento (mai avuto finora), per il criterio già discusso in
    chat: "rendimento non troppo penalizzato rispetto a V6 pura, con
    drawdown migliore".

Router combinato: in PAUSA (18/07/2026), non incluso qui.
V6: motore INVARIATO (BacktestEngineFloatingKillSwitch, forza la size
minima come da RCA sez.15 — decisione che resta ferma, non rimessa in
discussione da questo script).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 90
REAL_CAPITAL = 2000.0
FEASIBILITY_DAYS_BACK = 180
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
SPLITS = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]

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


def compute_drawdown(trades_df: pd.DataFrame, capital0: float) -> tuple[float, float]:
    if trades_df.empty:
        return 0.0, 0.0
    trades_sorted = trades_df.sort_values("exit_time")
    equity = capital0 + trades_sorted["pnl"].cumsum()
    running_max = equity.cummax()
    dd_eur = equity - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "max_dd_pct": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    dd_pct, _ = compute_drawdown(trades_df, capital0)
    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(), "max_dd_pct": dd_pct}


def skip_stats(engine_) -> int:
    return getattr(engine_, "n_skipped_min_size", 0)


def run_phase1_feasibility() -> str:
    print("=" * 70)
    print("FASE 1 — Fattibilità corretta (motore che salta, non forza)")
    print("=" * 70)

    end = datetime.now()
    start = end - timedelta(days=FEASIBILITY_DAYS_BACK)
    warmup_start = start - timedelta(days=WARMUP_DAYS)

    raw_data = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name}...")
        raw_data[name] = fetch_bars(const, warmup_start, end)

    results = {}
    for mode in ("bollinger", "rsi"):
        signal_data = {name: generate_mean_reversion_signals(raw_data[name], eng.INSTRUMENTS[name], mode=mode)
                       for name in SYMBOLS}
        engine_ = BacktestEngineMeanReversion(capital0=REAL_CAPITAL)
        trades, _ = engine_.run(signal_data)
        m = metrics_summary(trades, REAL_CAPITAL)
        n_skipped = skip_stats(engine_)

        print(f"\n--- {mode.upper()} ---")
        for instr in SYMBOLS:
            n_instr = (trades["instrument"] == instr).sum() if not trades.empty else 0
            print(f"  {instr}: {n_instr} trade")
        print(f"  Totale: n={m['n_trades']} WR={m['win_rate_pct']:.1f}% PF={m['profit_factor']:.2f} "
              f"PnL={m['pnl_total']:+.2f} MaxDD={m['max_dd_pct']:.1f}%")
        print(f"  Trade SALTATI per size insufficiente: {n_skipped} "
              f"(prima venivano forzati — questo è il costo della correzione)")

        results[mode] = m

    selected = "rsi" if results["rsi"]["profit_factor"] >= results["bollinger"]["profit_factor"] else "bollinger"
    print(f"\n>>> Variante selezionata per FASE 2: {selected.upper()} "
          f"(PF {results[selected]['profit_factor']:.2f} vs "
          f"{results['bollinger' if selected == 'rsi' else 'rsi']['profit_factor']:.2f} dell'altra)")

    return selected


def get_period_raw(period_start: str, period_end: str):
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)
    raw = {}
    for name, const in SYMBOLS.items():
        raw[name] = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
    return raw, p_start


def slice_from(signals_df: pd.DataFrame, p_start: pd.Timestamp) -> pd.DataFrame:
    return signals_df[signals_df["timestamp"] >= p_start].reset_index(drop=True)


def run_phase2_split_test(mr_mode: str):
    print("\n" + "=" * 70)
    print(f"FASE 2 — Test split capitale reale ({REAL_CAPITAL:.0f} EUR), mean-reversion={mr_mode}")
    print("=" * 70)

    all_rows = []
    for label, p_start_str, p_end_str in PERIODS:
        print(f"\n{'-'*70}\nPeriodo {label}\n{'-'*70}")
        raw_data, p_start = get_period_raw(p_start_str, p_end_str)

        v6_signal_data = {name: slice_from(eng.generate_signals(raw_data[name], eng.INSTRUMENTS[name]), p_start)
                           for name in SYMBOLS}
        mr_signal_data = {name: slice_from(generate_mean_reversion_signals(raw_data[name], eng.INSTRUMENTS[name], mode=mr_mode), p_start)
                           for name in SYMBOLS}

        engine_v6_pure = BacktestEngineFloatingKillSwitch(capital0=REAL_CAPITAL)
        trades_v6_pure, _ = engine_v6_pure.run(v6_signal_data)
        m_v6_pure = metrics_summary(trades_v6_pure, REAL_CAPITAL)
        print(f"  V6 PURA (2000 EUR interi): rendimento={100*m_v6_pure['pnl_total']/REAL_CAPITAL:+.1f}% "
              f"MaxDD={m_v6_pure['max_dd_pct']:.1f}%")

        for v6_pct, mr_pct in SPLITS:
            cap_v6 = REAL_CAPITAL * v6_pct
            cap_mr = REAL_CAPITAL * mr_pct

            engine_v6 = BacktestEngineFloatingKillSwitch(capital0=cap_v6)
            trades_v6, _ = engine_v6.run(v6_signal_data)

            engine_mr = BacktestEngineMeanReversion(capital0=cap_mr)
            trades_mr, _ = engine_mr.run(mr_signal_data)

            m_v6 = metrics_summary(trades_v6, cap_v6)
            m_mr = metrics_summary(trades_mr, cap_mr)
            n_skipped = skip_stats(engine_mr)

            total_pnl = m_v6["pnl_total"] + m_mr["pnl_total"]
            total_ret_pct = 100 * total_pnl / REAL_CAPITAL

            events = []
            for _, t in trades_v6.iterrows():
                events.append((pd.Timestamp(t["exit_time"]), "v6", t["pnl"]))
            for _, t in trades_mr.iterrows():
                events.append((pd.Timestamp(t["exit_time"]), "mr", t["pnl"]))
            events.sort(key=lambda e: e[0])
            cv6, cmr = cap_v6, cap_mr
            totals = []
            for _, which, pnl in events:
                if which == "v6":
                    cv6 += pnl
                else:
                    cmr += pnl
                totals.append(cv6 + cmr)
            if totals:
                totals_s = pd.Series(totals)
                dd_pct = ((totals_s - totals_s.cummax()) / totals_s.cummax()).min() * 100
            else:
                dd_pct = 0.0

            print(f"  Split {int(v6_pct*100)}/{int(mr_pct*100)}: rendimento={total_ret_pct:+.1f}% "
                  f"MaxDD={dd_pct:.1f}%  (MR: {m_mr['n_trades']} trade, {n_skipped} saltati per size)")

            all_rows.append({
                "periodo": label, "split_v6_pct": v6_pct, "split_mr_pct": mr_pct,
                "v6_pura_rendimento_pct": 100 * m_v6_pure["pnl_total"] / REAL_CAPITAL,
                "v6_pura_max_dd_pct": m_v6_pure["max_dd_pct"],
                "rendimento_totale_pct": total_ret_pct, "max_dd_pct": dd_pct,
                "mr_n_trades": m_mr["n_trades"], "mr_n_skipped_min_size": n_skipped,
            })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("mean_reversion_full_pipeline_results.csv", index=False)

    print(f"\n{'='*70}\nRIEPILOGO FINALE — medie sui 5 periodi\n{'='*70}")
    v6_pura_avg_ret = summary_df.groupby("periodo")["v6_pura_rendimento_pct"].first().mean()
    v6_pura_avg_dd = summary_df.groupby("periodo")["v6_pura_max_dd_pct"].first().mean()
    print(f"V6 pura (riferimento): rendimento medio {v6_pura_avg_ret:+.1f}%  MaxDD medio {v6_pura_avg_dd:.1f}%\n")

    for v6_pct, mr_pct in SPLITS:
        sub = summary_df[summary_df["split_v6_pct"] == v6_pct]
        ret = sub["rendimento_totale_pct"].mean()
        dd = sub["max_dd_pct"].mean()
        pct_of_v6 = 100 * ret / v6_pura_avg_ret if v6_pura_avg_ret != 0 else np.nan
        dd_improvement = v6_pura_avg_dd - dd
        print(f"  Split {int(v6_pct*100)}/{int(mr_pct*100)}: rendimento medio {ret:+.1f}% "
              f"({pct_of_v6:.0f}% di V6 pura)  MaxDD medio {dd:.1f}% "
              f"({dd_improvement:+.1f}pt vs V6 pura)")

    print("\nFile: mean_reversion_full_pipeline_results.csv")


def main():
    selected_mode = run_phase1_feasibility()
    run_phase2_split_test(selected_mode)


if __name__ == "__main__":
    main()
