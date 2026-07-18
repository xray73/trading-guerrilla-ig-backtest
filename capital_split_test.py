"""
capital_split_test.py — Testa diverse proporzioni di split del
capitale REALE (2.000 EUR, non 4.000 come nel test "separati" fatto
finora con due pool indipendenti da 2.000 EUR ciascuno) tra Variante 6
e mean-reversion, split FISSO (non dinamico — vedi motivazione in
chat 18/07/2026: un meccanismo che sposta capitale in base al momento
introdurrebbe un nuovo parametro da calibrare, rischio overfitting
già visto con altri meccanismi condizionali in questo progetto).

Per ciascuno split, su ciascuno dei 5 periodi ufficiali, riporta:
  - rendimento e drawdown (come nei test precedenti)
  - QUANTE VOLTE la size minima negoziabile (0.50) viene forzata,
    PER STRUMENTO — rilevante perché il DAX (rischio 2%, ma punto
    caro, ~24.800) è il candidato più esposto quando lo split lascia
    al mean-reversion poco capitale: il rischio reale per trade sale
    sopra quello previsto quando la size viene forzata al minimo.
  - quanti trade genera il mean-reversion per strumento (DAX vs
    FTSE100) — già visto nel test di fattibilità che FTSE100 ha un
    edge più forte, utile vedere se si conferma qui.

Split testati: (V6%, MR%) — 90/10, 80/20, 70/30, 60/40, 50/50.
"""

from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 90
REAL_CAPITAL = 2000.0   # il vincolo VERO, non 4.000 come nel test precedente
MR_MODE = "rsi"
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


def compute_drawdown(trades_df: pd.DataFrame, capital0: float) -> tuple[float, float]:
    if trades_df.empty:
        return 0.0, 0.0
    trades_sorted = trades_df.sort_values("exit_time")
    equity = capital0 + trades_sorted["pnl"].cumsum()
    running_max = equity.cummax()
    dd_eur = equity - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def forced_min_size_stats(trades_df: pd.DataFrame) -> dict:
    """% di trade con size forzata al minimo, per strumento — indica
    quanto spesso il rischio reale ha superato quello nominale."""
    out = {}
    for instr in SYMBOLS:
        sub = trades_df[trades_df["instrument"] == instr]
        if len(sub) == 0:
            out[instr] = {"n_trades": 0, "n_forced": 0, "pct_forced": np.nan}
            continue
        n_forced = sub["forced_min_size"].sum() if "forced_min_size" in sub.columns else 0
        out[instr] = {"n_trades": len(sub), "n_forced": int(n_forced),
                       "pct_forced": 100 * n_forced / len(sub)}
    return out


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n_trades": 0, "pnl_total": 0.0, "max_dd_pct": np.nan}
    dd_pct, _ = compute_drawdown(trades_df, capital0)
    return {"n_trades": n, "pnl_total": trades_df["pnl"].sum(), "max_dd_pct": dd_pct}


def main():
    print(f"=== Test split capitale reale ({REAL_CAPITAL:.0f} EUR) — V6 + mean-reversion ({MR_MODE}) ===\n")

    all_rows = []
    for label, p_start_str, p_end_str in PERIODS:
        print(f"\n{'='*70}\nPeriodo {label}\n{'='*70}")
        raw_data, p_start = get_period_raw(p_start_str, p_end_str)

        v6_signal_data = {name: slice_from(eng.generate_signals(raw_data[name], eng.INSTRUMENTS[name]), p_start)
                           for name in SYMBOLS}
        mr_signal_data = {name: slice_from(generate_mean_reversion_signals(raw_data[name], eng.INSTRUMENTS[name], mode=MR_MODE), p_start)
                           for name in SYMBOLS}

        for v6_pct, mr_pct in SPLITS:
            cap_v6 = REAL_CAPITAL * v6_pct
            cap_mr = REAL_CAPITAL * mr_pct

            engine_v6 = BacktestEngineFloatingKillSwitch(capital0=cap_v6)
            trades_v6, _ = engine_v6.run(v6_signal_data)

            engine_mr = BacktestEngineFloatingKillSwitch(capital0=cap_mr)
            trades_mr, _ = engine_mr.run(mr_signal_data)

            m_v6 = metrics_summary(trades_v6, cap_v6)
            m_mr = metrics_summary(trades_mr, cap_mr)

            total_pnl = m_v6["pnl_total"] + m_mr["pnl_total"]
            total_ret_pct = 100 * total_pnl / REAL_CAPITAL

            forced_v6 = forced_min_size_stats(trades_v6)
            forced_mr = forced_min_size_stats(trades_mr)

            print(f"\n  Split {int(v6_pct*100)}/{int(mr_pct*100)} (V6={cap_v6:.0f} EUR, MR={cap_mr:.0f} EUR):")
            print(f"    Rendimento totale: {total_ret_pct:+.1f}%  (V6: {m_v6['pnl_total']:+.2f}, MR: {m_mr['pnl_total']:+.2f})")
            print(f"    Trade MR per strumento: DAX={forced_mr['DAX']['n_trades']} "
                  f"(size forzata {forced_mr['DAX']['pct_forced']:.0f}%)  "
                  f"FTSE100={forced_mr['FTSE100']['n_trades']} (size forzata {forced_mr['FTSE100']['pct_forced']:.0f}%)")
            print(f"    Trade V6 per strumento: DAX size forzata {forced_v6['DAX']['pct_forced']:.0f}%  "
                  f"FTSE100 size forzata {forced_v6['FTSE100']['pct_forced']:.0f}%")

            all_rows.append({
                "periodo": label, "split_v6_pct": v6_pct, "split_mr_pct": mr_pct,
                "cap_v6": cap_v6, "cap_mr": cap_mr,
                "rendimento_totale_pct": total_ret_pct,
                "v6_pnl": m_v6["pnl_total"], "mr_pnl": m_mr["pnl_total"],
                "mr_dax_n_trades": forced_mr["DAX"]["n_trades"], "mr_dax_pct_forced": forced_mr["DAX"]["pct_forced"],
                "mr_ftse_n_trades": forced_mr["FTSE100"]["n_trades"], "mr_ftse_pct_forced": forced_mr["FTSE100"]["pct_forced"],
                "v6_dax_pct_forced": forced_v6["DAX"]["pct_forced"], "v6_ftse_pct_forced": forced_v6["FTSE100"]["pct_forced"],
            })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("capital_split_results.csv", index=False)

    print(f"\n{'='*70}\nRIEPILOGO — medie sui 5 periodi, per split\n{'='*70}")
    for v6_pct, mr_pct in SPLITS:
        sub = summary_df[(summary_df["split_v6_pct"] == v6_pct)]
        print(f"  Split {int(v6_pct*100)}/{int(mr_pct*100)}: rendimento medio {sub['rendimento_totale_pct'].mean():+.1f}%  "
              f"MR size forzata DAX media {sub['mr_dax_pct_forced'].mean():.0f}%  "
              f"FTSE100 media {sub['mr_ftse_pct_forced'].mean():.0f}%")

    print("\nFile: capital_split_results.csv")


if __name__ == "__main__":
    main()
