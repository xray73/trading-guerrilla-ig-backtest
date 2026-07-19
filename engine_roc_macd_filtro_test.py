"""
engine_roc_macd_filtro_test.py — Sanity check + test di impatto del
filtro di conferma momentum ROC+MACD (engine_roc_macd_filtro.py) sui 5
periodi ufficiali, capitale 2.000EUR, spread di produzione (DAX
1.2pt/FTSE100 1.0pt — stesso baseline usato per gli altri filtri
testati in questa sessione, per confronto diretto con quei risultati).

Check A (sanity): enabled=False deve produrre risultati IDENTICI al
motore standard (BacktestEngineFloatingKillSwitch) — bypass completo
del filtro, non solo soglie irraggiungibili (qui non ci sono soglie
libere da spingere all'infinito, la regola è binaria).

Check B (impatto): confronto baseline (enabled=False, cioè motore
standard) vs filtrato (enabled=True) sui 5 periodi ufficiali, tutte le
metriche richieste: n_trade, win_rate, profit_factor, PnL, max_drawdown,
expectancy per trade, quanti ingressi bloccati dal filtro.

Nessuna scrittura su D1. Nessuna modifica a engine.py, live_execute.py
o alle altre sottoclassi esistenti.
"""

from __future__ import annotations

import os
from datetime import timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_roc_macd_filtro import BacktestEngineROCMACDFiltro

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
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


def get_period_signal_data(period_start: str, period_end: str) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    signal_data = {}
    for name, const in SYMBOLS.items():
        raw = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        inst = eng.INSTRUMENTS[name]
        full_signals = eng.generate_signals(raw, inst)
        signal_data[name] = full_signals[full_signals["timestamp"] >= p_start].reset_index(drop=True)
    return signal_data


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "expectancy": np.nan, "max_drawdown_pct": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf

    equity = CAPITAL0 + trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max
    max_dd = drawdown_pct.min() * 100

    return {
        "n_trades": len(trades_df), "win_rate_pct": 100 * len(wins) / len(trades_df),
        "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(),
        "expectancy": trades_df["pnl"].mean(), "max_drawdown_pct": max_dd,
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Filtro conferma momentum ROC+MACD — sanity check + impatto 5 periodi ===\n")

    # --- Check A: sanity, stesso periodo/strumento piccolo per velocita' ---
    log("--- Check A: sanity (enabled=False deve == motore standard) ---")
    p_start, p_end = "2023-01-02", "2023-12-30"
    signal_data = get_period_signal_data(p_start, p_end)

    engine_std = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_std, _ = engine_std.run(signal_data)

    engine_bypass = BacktestEngineROCMACDFiltro(capital0=CAPITAL0, enabled=False)
    trades_bypass, _ = engine_bypass.run(signal_data)

    identical = (len(trades_std) == len(trades_bypass) and
                 abs(trades_std["pnl"].sum() - trades_bypass["pnl"].sum()) < 0.01)
    log(f"  Motore standard: n_trade={len(trades_std)} PnL={trades_std['pnl'].sum():+.2f}")
    log(f"  Bypass (enabled=False): n_trade={len(trades_bypass)} PnL={trades_bypass['pnl'].sum():+.2f}")
    log(f"  Check A: {'PASS' if identical else 'FAIL - INDAGARE PRIMA DI PROCEDERE'}\n")

    if not identical:
        log("ERRORE: sanity check fallito, il resto del test non è affidabile. Fermo qui.")
        with open("results/engine_roc_macd_filtro_test.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # --- Check B: impatto sui 5 periodi ufficiali ---
    log("--- Check B: impatto sui 5 periodi ufficiali (baseline vs filtrato) ---")
    rows = []
    for label, p_start, p_end in PERIODS:
        log(f"\n  Periodo {label}")
        signal_data = get_period_signal_data(p_start, p_end)

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_base, _ = eng_base.run(signal_data)
        m_base = metrics_summary(trades_base)

        eng_filt = BacktestEngineROCMACDFiltro(capital0=CAPITAL0, enabled=True)
        trades_filt, _ = eng_filt.run(signal_data)
        m_filt = metrics_summary(trades_filt)

        n_blocked = sum(eng_filt.n_blocked.values())

        log(f"    Baseline:  n={m_base['n_trades']:3d}  WR={m_base['win_rate_pct']:.1f}%  "
            f"PF={m_base['profit_factor']:.2f}  PnL={m_base['pnl_total']:+.2f}  "
            f"maxDD={m_base['max_drawdown_pct']:.1f}%  exp={m_base['expectancy']:+.2f}")
        log(f"    Filtrato:  n={m_filt['n_trades']:3d}  WR={m_filt['win_rate_pct']:.1f}%  "
            f"PF={m_filt['profit_factor']:.2f}  PnL={m_filt['pnl_total']:+.2f}  "
            f"maxDD={m_filt['max_drawdown_pct']:.1f}%  exp={m_filt['expectancy']:+.2f}")
        log(f"    Ingressi bloccati dal filtro: {n_blocked}  "
            f"(DAX={eng_filt.n_blocked.get('DAX',0)}, FTSE100={eng_filt.n_blocked.get('FTSE100',0)})")
        log(f"    Delta PnL: {m_filt['pnl_total'] - m_base['pnl_total']:+.2f}  "
            f"Delta WR: {m_filt['win_rate_pct'] - m_base['win_rate_pct']:+.1f}pt")

        rows.append({
            "periodo": label,
            "base_n": m_base["n_trades"], "base_wr": m_base["win_rate_pct"],
            "base_pf": m_base["profit_factor"], "base_pnl": m_base["pnl_total"],
            "base_dd": m_base["max_drawdown_pct"], "base_exp": m_base["expectancy"],
            "filt_n": m_filt["n_trades"], "filt_wr": m_filt["win_rate_pct"],
            "filt_pf": m_filt["profit_factor"], "filt_pnl": m_filt["pnl_total"],
            "filt_dd": m_filt["max_drawdown_pct"], "filt_exp": m_filt["expectancy"],
            "n_blocked": n_blocked,
        })

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/engine_roc_macd_filtro_test.csv", index=False)

    log(f"\n{'='*70}\nRIEPILOGO — somma/media sui 5 periodi ufficiali\n{'='*70}")
    log(f"PnL totale — baseline: {summary_df['base_pnl'].sum():+.2f}  "
        f"filtrato: {summary_df['filt_pnl'].sum():+.2f}  "
        f"delta: {summary_df['filt_pnl'].sum() - summary_df['base_pnl'].sum():+.2f}")
    log(f"Trade totali — baseline: {summary_df['base_n'].sum()}  filtrato: {summary_df['filt_n'].sum()}  "
        f"(bloccati: {summary_df['n_blocked'].sum()})")
    log(f"Win rate medio — baseline: {summary_df['base_wr'].mean():.1f}%  "
        f"filtrato: {summary_df['filt_wr'].mean():.1f}%")
    log(f"PF medio — baseline: {summary_df['base_pf'].replace([np.inf], np.nan).mean():.2f}  "
        f"filtrato: {summary_df['filt_pf'].replace([np.inf], np.nan).mean():.2f}")
    log(f"Max drawdown medio — baseline: {summary_df['base_dd'].mean():.1f}%  "
        f"filtrato: {summary_df['filt_dd'].mean():.1f}%")
    log(f"Periodi in cui il filtro migliora il PnL: "
        f"{(summary_df['filt_pnl'] > summary_df['base_pnl']).sum()}/5")

    with open("results/engine_roc_macd_filtro_test.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
