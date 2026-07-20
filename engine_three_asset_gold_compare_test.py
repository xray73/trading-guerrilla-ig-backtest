"""
engine_three_asset_gold_compare_test.py — Confronto a 3 vie sui 5
periodi ufficiali:
  1) Baseline (DAX+FTSE100, motore standard)
  2) +GOLD, selezione a sola correlazione (engine_three_asset_gold.py)
  3) +GOLD, selezione a correlazione CON pavimento ADX relativo
     (engine_three_asset_gold_adxfloor.py)

Motivazione (19/07/2026): l'isolamento (analyze_gold_isolation.py) ha
mostrato che la selezione a sola correlazione NON è un problema
sistemico (PnL ombra dei trade esclusi -452,58EUR aggregato, cioè i
trade scalzati erano in media deboli) — MA c'è un'eccezione precisa,
2020-covid (PnL ombra +115,73, unico periodo dove GOLD peggiora,
delta -261,30). Il pavimento ADX dovrebbe correggere selettivamente
questo caso senza intaccare il guadagno netto degli altri 4 periodi.

Sanity check: con GOLD escluso, la variante ADX-floor deve produrre
risultati IDENTICI al motore standard (stessa convenzione delle altre
sottoclassi).

Nessuna scrittura su D1. Nessuna modifica a engine.py o alle altre
sottoclassi esistenti.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_three_asset_gold import BacktestEngineV6Gold, instruments_with_gold
from engine_three_asset_gold_adxfloor import BacktestEngineV6GoldADXFloor
from ohlc_data_source import get_ohlc

CAPITAL_V6 = 1400.0
SYMBOLS_3 = ["DAX", "FTSE100", "GOLD"]

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "max_drawdown_pct": 0.0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    equity = capital0 + trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    return {
        "n_trades": len(trades_df), "win_rate_pct": 100 * len(wins) / len(trades_df),
        "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(), "max_drawdown_pct": dd,
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Confronto a 3 vie: baseline / +GOLD correlazione pura / +GOLD correlazione+pavimento ADX ===\n")

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    instruments_2 = dict(eng.INSTRUMENTS)
    instruments_3 = instruments_with_gold()

    log("Verifico/aggiorno OHLC (D1 + eventuali barre mancanti da Dukascopy)...")
    raw_full = {name: get_ohlc(name, account_id, token, log=log) for name in SYMBOLS_3}
    for name, df in raw_full.items():
        log(f"  {name}: {len(df)} righe, fino a {df['timestamp'].max()}")
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(raw_full[name], instruments_3[name]) for name in SYMBOLS_3}

    # ============================================================
    # Sanity check: ADX-floor con GOLD escluso == motore standard
    # ============================================================
    log("--- Sanity check: ADX-floor (GOLD escluso) deve == motore standard, periodo 2023 ---")
    ps = pd.Timestamp("2023-01-02", tz="UTC")
    pe = pd.Timestamp("2023-12-30", tz="UTC") + pd.Timedelta(days=1)
    sig_2023_2 = {k: slice_period(v6_signals_full[k], ps, pe) for k in ("DAX", "FTSE100")}

    eng_std = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_std, _ = eng_std.run(sig_2023_2)

    eng_adxfloor_bypass = BacktestEngineV6GoldADXFloor(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_bypass, _ = eng_adxfloor_bypass.run(sig_2023_2)

    identical = (len(trades_std) == len(trades_bypass) and
                 abs(trades_std["pnl"].sum() - trades_bypass["pnl"].sum()) < 0.01)
    log(f"  Standard: n={len(trades_std)} PnL={trades_std['pnl'].sum():+.2f}")
    log(f"  ADX-floor (GOLD escluso): n={len(trades_bypass)} PnL={trades_bypass['pnl'].sum():+.2f}")
    log(f"  Sanity check: {'PASS' if identical else 'FAIL - fermo qui'}\n")

    if not identical:
        os.makedirs("results", exist_ok=True)
        with open("results/engine_three_asset_gold_compare_test.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # ============================================================
    # Confronto a 3 vie sui 5 periodi
    # ============================================================
    rows = []
    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        p_end = pd.Timestamp(p_end_str, tz="UTC") + pd.Timedelta(days=1)
        log(f"Periodo {label}")

        sig_3 = {name: slice_period(v6_signals_full[name], p_start, p_end) for name in SYMBOLS_3}
        sig_2 = {k: v for k, v in sig_3.items() if k != "GOLD"}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
        trades_base, _ = eng_base.run(sig_2)
        m_base = metrics_summary(trades_base, CAPITAL_V6)

        eng_corr = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_corr, _ = eng_corr.run(sig_3)
        m_corr = metrics_summary(trades_corr, CAPITAL_V6)

        eng_adxfloor = BacktestEngineV6GoldADXFloor(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_adxfloor, _ = eng_adxfloor.run(sig_3)
        m_adxfloor = metrics_summary(trades_adxfloor, CAPITAL_V6)

        log(f"  Baseline:              n={m_base['n_trades']:3d}  WR={m_base['win_rate_pct']:.1f}%  "
            f"PF={m_base['profit_factor']:.2f}  PnL={m_base['pnl_total']:+.2f}  maxDD={m_base['max_drawdown_pct']:.1f}%")
        log(f"  +GOLD corr. pura:      n={m_corr['n_trades']:3d}  WR={m_corr['win_rate_pct']:.1f}%  "
            f"PF={m_corr['profit_factor']:.2f}  PnL={m_corr['pnl_total']:+.2f}  maxDD={m_corr['max_drawdown_pct']:.1f}%  "
            f"(delta base: {m_corr['pnl_total']-m_base['pnl_total']:+.2f})")
        log(f"  +GOLD corr.+ADX-floor: n={m_adxfloor['n_trades']:3d}  WR={m_adxfloor['win_rate_pct']:.1f}%  "
            f"PF={m_adxfloor['profit_factor']:.2f}  PnL={m_adxfloor['pnl_total']:+.2f}  maxDD={m_adxfloor['max_drawdown_pct']:.1f}%  "
            f"(delta base: {m_adxfloor['pnl_total']-m_base['pnl_total']:+.2f}, "
            f"delta corr.pura: {m_adxfloor['pnl_total']-m_corr['pnl_total']:+.2f})\n")

        rows.append({
            "periodo": label,
            "base_n": m_base["n_trades"], "base_pnl": m_base["pnl_total"], "base_dd": m_base["max_drawdown_pct"],
            "corr_n": m_corr["n_trades"], "corr_pnl": m_corr["pnl_total"], "corr_dd": m_corr["max_drawdown_pct"],
            "adxfloor_n": m_adxfloor["n_trades"], "adxfloor_pnl": m_adxfloor["pnl_total"],
            "adxfloor_dd": m_adxfloor["max_drawdown_pct"],
        })

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/engine_three_asset_gold_compare_test.csv", index=False)

    log(f"{'='*70}\nRIEPILOGO — somma sui 5 periodi ufficiali\n{'='*70}")
    log(f"Baseline:              PnL={summary_df['base_pnl'].sum():+.2f}")
    log(f"+GOLD corr. pura:      PnL={summary_df['corr_pnl'].sum():+.2f}  "
        f"delta={summary_df['corr_pnl'].sum()-summary_df['base_pnl'].sum():+.2f}  "
        f"DD medio={summary_df['corr_dd'].mean():.1f}%")
    log(f"+GOLD corr.+ADX-floor: PnL={summary_df['adxfloor_pnl'].sum():+.2f}  "
        f"delta={summary_df['adxfloor_pnl'].sum()-summary_df['base_pnl'].sum():+.2f}  "
        f"DD medio={summary_df['adxfloor_dd'].mean():.1f}%")
    log(f"\nPeriodi dove ADX-floor migliora rispetto a correlazione pura: "
        f"{(summary_df['adxfloor_pnl'] > summary_df['corr_pnl']).sum()}/5")
    log(f"Periodi dove ADX-floor peggiora rispetto a correlazione pura: "
        f"{(summary_df['adxfloor_pnl'] < summary_df['corr_pnl']).sum()}/5")
    covid_row = summary_df[summary_df["periodo"] == "2020-covid"].iloc[0]
    log(f"\n2020-covid (il caso che doveva correggere): corr.pura={covid_row['corr_pnl']:+.2f}  "
        f"ADX-floor={covid_row['adxfloor_pnl']:+.2f}  baseline={covid_row['base_pnl']:+.2f}")

    with open("results/engine_three_asset_gold_compare_test.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
