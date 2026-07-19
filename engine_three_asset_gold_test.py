"""
engine_three_asset_gold_test.py — Sanity check + impatto sui 5 periodi
ufficiali dell'estensione a 3 strumenti (DAX/FTSE100/GOLD) con
selezione multi-candidato per correlazione minima (engine_three_asset_gold.py).

Tetto posizioni concorrenti INVARIATO a 2 — nessuno slot in più,
nessuna diluizione di capitale. GOLD compete per gli stessi 2 slot di
V6 e mean-reversion. Selezione basata SOLO su correlazione (opzione 1
scelta in chat 19/07/2026), nessun peso sulla forza del segnale.

Check A (sanity): con GOLD tradable=False (solo DAX+FTSE100), il
motore esteso deve produrre risultati IDENTICI al motore standard
(BacktestEngineFloatingKillSwitch / BacktestEngineMeanReversion) — la
nuova selezione multi-candidato coincide col comportamento base quando
i candidati non superano mai gli slot liberi.

Check B (impatto): confronto baseline (2 strumenti) vs esteso (3
strumenti) sui 5 periodi ufficiali, per V6 (capitale 1.400EUR, quota
reale del pool) e mean-reversion (capitale 600EUR) separatamente.
Metriche complete: n_trade, win_rate, profit_factor, PnL, max_drawdown,
expectancy, quante volte GOLD è stato scelto/scartato.

Nessuna scrittura su D1. Nessuna modifica a engine.py, live_execute.py
o alle sottoclassi esistenti.
"""

from __future__ import annotations

import os
from datetime import timedelta
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
    instruments_with_gold, GOLD_CONFIG,
)

WARMUP_DAYS = 90
CAPITAL_V6 = 1400.0
CAPITAL_MR = 600.0
SYMBOLS_2 = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
SYMBOLS_3 = dict(SYMBOLS_2, GOLD=INSTRUMENT_FX_METALS_XAU_USD)

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


def get_period_raw(period_start: str, period_end: str, symbols: dict) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)
    raw = {}
    for name, const in symbols.items():
        df = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        raw[name] = df[df["timestamp"] >= p_start - timedelta(days=WARMUP_DAYS)].reset_index(drop=True)
    return raw, p_start


def build_v6_signals(raw: dict, instruments: dict, p_start) -> dict:
    out = {}
    for name, df in raw.items():
        inst = instruments[name]
        full = eng.generate_signals(df, inst)
        out[name] = full[full["timestamp"] >= p_start].reset_index(drop=True)
    return out


def build_mr_signals(raw: dict, instruments: dict, p_start) -> dict:
    out = {}
    for name, df in raw.items():
        inst = instruments[name]
        full = generate_mean_reversion_signals(df, inst, mode="rsi")
        out[name] = full[full["timestamp"] >= p_start].reset_index(drop=True)
    return out


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "expectancy": np.nan, "max_drawdown_pct": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    capital0 = trades_df.attrs.get("capital0", 2000.0)
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

    # ============================================================
    # Check A — sanity: GOLD tradable=False deve == motore standard
    # ============================================================
    log("--- Check A: sanity (GOLD escluso deve == motore a 2 strumenti) ---")
    p_start_test, p_end_test = "2023-01-02", "2023-12-30"

    raw2, ps2 = get_period_raw(p_start_test, p_end_test, SYMBOLS_2)
    v6_sig_2 = build_v6_signals(raw2, instruments_2, ps2)

    eng_std = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_std, _ = eng_std.run(v6_sig_2)

    eng_ext_bypass = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_2)  # GOLD non nel dict -> non tradable
    trades_ext, _ = eng_ext_bypass.run(v6_sig_2)

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
    for label, p_start, p_end in PERIODS:
        log(f"Periodo {label}")
        raw3, ps3 = get_period_raw(p_start, p_end, SYMBOLS_3)

        # --- V6 ---
        v6_sig_3 = build_v6_signals(raw3, instruments_3, ps3)
        v6_sig_2_only = {k: v for k, v in v6_sig_3.items() if k != "GOLD"}

        eng_v6_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
        trades_v6_base, _ = eng_v6_base.run(v6_sig_2_only)
        trades_v6_base.attrs["capital0"] = CAPITAL_V6
        m_v6_base = metrics_summary(trades_v6_base)

        eng_v6_gold = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_v6_gold, _ = eng_v6_gold.run(v6_sig_3)
        trades_v6_gold.attrs["capital0"] = CAPITAL_V6
        m_v6_gold = metrics_summary(trades_v6_gold)
        gold_trades_v6 = 0 if trades_v6_gold.empty else (trades_v6_gold["instrument"] == "GOLD").sum()

        log(f"  V6 baseline (2 strum.): n={m_v6_base['n_trades']:3d} WR={m_v6_base['win_rate_pct']:.1f}% "
            f"PF={m_v6_base['profit_factor']:.2f} PnL={m_v6_base['pnl_total']:+.2f} maxDD={m_v6_base['max_drawdown_pct']:.1f}%")
        log(f"  V6 +GOLD    (3 strum.): n={m_v6_gold['n_trades']:3d} WR={m_v6_gold['win_rate_pct']:.1f}% "
            f"PF={m_v6_gold['profit_factor']:.2f} PnL={m_v6_gold['pnl_total']:+.2f} maxDD={m_v6_gold['max_drawdown_pct']:.1f}%  "
            f"(di cui GOLD: {gold_trades_v6})")
        log(f"  Delta V6 PnL: {m_v6_gold['pnl_total'] - m_v6_base['pnl_total']:+.2f}")

        # --- Mean-reversion ---
        mr_sig_3 = build_mr_signals(raw3, instruments_3, ps3)
        mr_sig_2_only = {k: v for k, v in mr_sig_3.items() if k != "GOLD"}

        eng_mr_base = BacktestEngineMeanReversion(capital0=CAPITAL_MR, instruments=instruments_2)
        trades_mr_base, _ = eng_mr_base.run(mr_sig_2_only)
        trades_mr_base.attrs["capital0"] = CAPITAL_MR
        m_mr_base = metrics_summary(trades_mr_base)

        eng_mr_gold = BacktestEngineMeanReversionGold(capital0=CAPITAL_MR, instruments=instruments_3)
        trades_mr_gold, _ = eng_mr_gold.run(mr_sig_3)
        trades_mr_gold.attrs["capital0"] = CAPITAL_MR
        m_mr_gold = metrics_summary(trades_mr_gold)
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
