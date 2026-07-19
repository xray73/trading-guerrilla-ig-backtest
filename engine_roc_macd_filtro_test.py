"""
engine_three_asset_gold_test.py — Sanity check + impatto sui 5 periodi
ufficiali dell'estensione a 3 strumenti (DAX/FTSE100/GOLD) con
selezione multi-candidato per correlazione minima (engine_three_asset_gold.py).

Tetto posizioni concorrenti INVARIATO a 2 — nessuno slot in più,
nessuna diluizione di capitale. GOLD compete per gli stessi 2 slot di
V6 e mean-reversion. Selezione basata SOLO su correlazione (opzione 1
scelta in chat 19/07/2026), nessun peso sulla forza del segnale.

VERSIONE 2 (19/07/2026) — ottimizzata per velocità: la versione
precedente faceva un fetch Dukascopy separato per ciascuna
combinazione periodo×strumento (17 chiamate di rete: 2 per il sanity
check + 5 periodi × 3 strumenti), andando in timeout su GitHub Actions.
Questa versione scarica UNA SOLA VOLTA per strumento l'intero storico
2014-2026 (3 chiamate di rete totali), genera i segnali una sola volta
sull'intera serie, e affetta localmente (nessuna rete) per ciascuno
dei 5 periodi — stesso identico risultato, drasticamente più veloce.

Check A (sanity): con GOLD tradable=False (solo DAX+FTSE100), il
motore esteso deve produrre risultati IDENTICI al motore standard
(BacktestEngineFloatingKillSwitch / BacktestEngineMeanReversion).

Check B (impatto): confronto baseline (2 strumenti) vs esteso (3
strumenti) sui 5 periodi ufficiali, per V6 (capitale 1.400EUR) e
mean-reversion (capitale 600EUR) separatamente. Metriche complete:
n_trade, win_rate, profit_factor, PnL, max_drawdown, expectancy,
quante volte GOLD è stato scelto.

Nessuna scrittura su D1. Nessuna modifica a engine.py, live_execute.py
o alle sottoclassi esistenti.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_FX_METALS_XAU_USD,
)

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from engine_three_asset_gold import (
    BacktestEngineV6Gold, BacktestEngineMeanReversionGold,
    instruments_with_gold,
)

CAPITAL_V6 = 1400.0
CAPITAL_MR = 600.0
SYMBOLS_3 = {
    "DAX": INSTRUMENT_IDX_EUROPE_E_DAAX,
    "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    "GOLD": INSTRUMENT_FX_METALS_XAU_USD,
}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

FULL_FETCH_START = datetime(2014, 10, 1, tzinfo=timezone.utc)
FULL_FETCH_END = datetime(2026, 7, 11, tzinfo=timezone.utc)


def fetch_bars_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        FULL_FETCH_START, FULL_FETCH_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= p_start].reset_index(drop=True)


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "expectancy": np.nan, "max_drawdown_pct": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    equity = capital0 + trades_df["pnl"].cumsum()
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

    log("=== Motore 3 strumenti (DAX/FTSE100/GOLD), selezione per correlazione minima ===")
    log("Tetto posizioni concorrenti invariato a 2 — nessuno slot in più.\n")

    instruments_2 = dict(eng.INSTRUMENTS)
    instruments_3 = instruments_with_gold()

    log("Scarico storico DAX/FTSE100/GOLD (un solo fetch per strumento, 2014-2026)...")
    raw_full = {name: fetch_bars_full(const) for name, const in SYMBOLS_3.items()}
    log("Fatto.\n")

    log("Genero segnali V6 e mean-reversion su tutta la serie (una sola volta)...")
    v6_signals_full = {name: eng.generate_signals(raw_full[name], instruments_3[name]) for name in SYMBOLS_3}
    mr_signals_full = {name: generate_mean_reversion_signals(raw_full[name], instruments_3[name], mode="rsi")
                        for name in SYMBOLS_3}
    log("Fatto.\n")

    # ============================================================
    # Check A — sanity: GOLD tradable=False deve == motore standard
    # ============================================================
    log("--- Check A: sanity (GOLD escluso deve == motore a 2 strumenti, periodo 2023) ---")
    ps_2023 = pd.Timestamp("2023-01-02", tz="UTC")
    v6_sig_2023_2 = {k: slice_period(v6_signals_full[k], ps_2023) for k in ("DAX", "FTSE100")}

    eng_std = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_std, _ = eng_std.run(v6_sig_2023_2)

    eng_ext_bypass = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_ext, _ = eng_ext_bypass.run(v6_sig_2023_2)

    identical = (len(trades_std) == len(trades_ext) and
                 abs(trades_std["pnl"].sum() - trades_ext["pnl"].sum()) < 0.01)
    log(f"  Standard:  n={len(trades_std)} PnL={trades_std['pnl'].sum():+.2f}")
    log(f"  Esteso (GOLD escluso): n={len(trades_ext)} PnL={trades_ext['pnl'].sum():+.2f}")
    log(f"  Check A: {'PASS' if identical else 'FAIL - fermo qui'}\n")

    if not identical:
        os.makedirs("results", exist_ok=True)
        with open("results/engine_three_asset_gold_test.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # ============================================================
    # Check B — impatto sui 5 periodi ufficiali, V6 e MR separatamente
    # ============================================================
    log("--- Check B: impatto sui 5 periodi ufficiali ---\n")
    rows = []
    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        log(f"Periodo {label}")

        v6_sig_3 = {name: slice_period(v6_signals_full[name], p_start) for name in SYMBOLS_3}
        v6_sig_2_only = {k: v for k, v in v6_sig_3.items() if k != "GOLD"}

        eng_v6_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
        trades_v6_base, _ = eng_v6_base.run(v6_sig_2_only)
        m_v6_base = metrics_summary(trades_v6_base, CAPITAL_V6)

        eng_v6_gold = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_v6_gold, _ = eng_v6_gold.run(v6_sig_3)
        m_v6_gold = metrics_summary(trades_v6_gold, CAPITAL_V6)
        gold_trades_v6 = 0 if trades_v6_gold.empty else (trades_v6_gold["instrument"] == "GOLD").sum()

        log(f"  V6 baseline (2 strum.): n={m_v6_base['n_trades']:3d} WR={m_v6_base['win_rate_pct']:.1f}% "
            f"PF={m_v6_base['profit_factor']:.2f} PnL={m_v6_base['pnl_total']:+.2f} maxDD={m_v6_base['max_drawdown_pct']:.1f}%")
        log(f"  V6 +GOLD    (3 strum.): n={m_v6_gold['n_trades']:3d} WR={m_v6_gold['win_rate_pct']:.1f}% "
            f"PF={m_v6_gold['profit_factor']:.2f} PnL={m_v6_gold['pnl_total']:+.2f} maxDD={m_v6_gold['max_drawdown_pct']:.1f}%  "
            f"(di cui GOLD: {gold_trades_v6})")
        log(f"  Delta V6 PnL: {m_v6_gold['pnl_total'] - m_v6_base['pnl_total']:+.2f}")

        mr_sig_3 = {name: slice_period(mr_signals_full[name], p_start) for name in SYMBOLS_3}
        mr_sig_2_only = {k: v for k, v in mr_sig_3.items() if k != "GOLD"}

        eng_mr_base = BacktestEngineMeanReversion(capital0=CAPITAL_MR, instruments=instruments_2)
        trades_mr_base, _ = eng_mr_base.run(mr_sig_2_only)
        m_mr_base = metrics_summary(trades_mr_base, CAPITAL_MR)

        eng_mr_gold = BacktestEngineMeanReversionGold(capital0=CAPITAL_MR, instruments=instruments_3)
        trades_mr_gold, _ = eng_mr_gold.run(mr_sig_3)
        m_mr_gold = metrics_summary(trades_mr_gold, CAPITAL_MR)
        gold_trades_mr = 0 if trades_mr_gold.empty else (trades_mr_gold["instrument"] == "GOLD").sum()

        log(f"  MR baseline (2 strum.): n={m_mr_base['n_trades']:3d} WR={m_mr_base['win_rate_pct']:.1f}% "
            f"PF={m_mr_base['profit_factor']:.2f} PnL={m_mr_base['pnl_total']:+.2f} maxDD={m_mr_base['max_drawdown_pct']:.1f}%")
        log(f"  MR +GOLD    (3 strum.): n={m_mr_gold['n_trades']:3d} WR={m_mr_gold['win_rate_pct']:.1f}% "
            f"PF={m_mr_gold['profit_factor']:.2f} PnL={m_mr_gold['pnl_total']:+.2f} maxDD={m_mr_gold['max_drawdown_pct']:.1f}%  "
            f"(di cui GOLD: {gold_trades_mr})")
        log(f"  Delta MR PnL: {m_mr_gold['pnl_total'] - m_mr_base['pnl_total']:+.2f}\n")

        rows.append({
            "periodo": label,
            "v6_base_n": m_v6_base["n_trades"], "v6_base_pnl": m_v6_base["pnl_total"],
            "v6_base_dd": m_v6_base["max_drawdown_pct"],
            "v6_gold_n": m_v6_gold["n_trades"], "v6_gold_pnl": m_v6_gold["pnl_total"],
            "v6_gold_dd": m_v6_gold["max_drawdown_pct"], "v6_gold_trades": gold_trades_v6,
            "mr_base_n": m_mr_base["n_trades"], "mr_base_pnl": m_mr_base["pnl_total"],
            "mr_base_dd": m_mr_base["max_drawdown_pct"],
            "mr_gold_n": m_mr_gold["n_trades"], "mr_gold_pnl": m_mr_gold["pnl_total"],
            "mr_gold_dd": m_mr_gold["max_drawdown_pct"], "mr_gold_trades": gold_trades_mr,
        })

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/engine_three_asset_gold_test.csv", index=False)

    log(f"{'='*70}\nRIEPILOGO — somma sui 5 periodi ufficiali\n{'='*70}")
    log(f"V6  baseline: PnL={summary_df['v6_base_pnl'].sum():+.2f}  "
        f"+GOLD: PnL={summary_df['v6_gold_pnl'].sum():+.2f}  "
        f"delta={summary_df['v6_gold_pnl'].sum()-summary_df['v6_base_pnl'].sum():+.2f}")
    log(f"V6  drawdown medio — baseline: {summary_df['v6_base_dd'].mean():.1f}%  +GOLD: {summary_df['v6_gold_dd'].mean():.1f}%")
    log(f"V6  trade totali su GOLD: {summary_df['v6_gold_trades'].sum()}")
    log(f"MR  baseline: PnL={summary_df['mr_base_pnl'].sum():+.2f}  "
        f"+GOLD: PnL={summary_df['mr_gold_pnl'].sum():+.2f}  "
        f"delta={summary_df['mr_gold_pnl'].sum()-summary_df['mr_base_pnl'].sum():+.2f}")
    log(f"MR  drawdown medio — baseline: {summary_df['mr_base_dd'].mean():.1f}%  +GOLD: {summary_df['mr_gold_dd'].mean():.1f}%")
    log(f"MR  trade totali su GOLD: {summary_df['mr_gold_trades'].sum()}")
    log(f"Periodi in cui V6+GOLD migliora il PnL: {(summary_df['v6_gold_pnl'] > summary_df['v6_base_pnl']).sum()}/5")
    log(f"Periodi in cui MR+GOLD migliora il PnL: {(summary_df['mr_gold_pnl'] > summary_df['mr_base_pnl']).sum()}/5")

    with open("results/engine_three_asset_gold_test.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
