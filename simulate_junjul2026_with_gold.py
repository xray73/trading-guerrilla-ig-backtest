"""
simulate_junjul2026_with_gold.py — Riusa esattamente lo stesso periodo
e la stessa diagnostica di diagnose_junjul2026_losses.py (1 giugno -
19 luglio 2026, il periodo con FTSE100 a win rate 10%, causa
identificata: shock geopolitico USA-Iran/petrolio, vedi chat 19/07/2026)
per rispondere alla domanda diretta: **con GOLD come terzo strumento
candidato (engine_three_asset_gold.py), come sarebbe cambiato quel
periodo specifico?**

Confronta, capitale reale di produzione (V6=1.400EUR, MR=600EUR):
  - V6 baseline (DAX+FTSE100) vs V6+GOLD (DAX+FTSE100+GOLD)
  - MR baseline (DAX+FTSE100) vs MR+GOLD (DAX+FTSE100+GOLD)

Stessa diagnostica di diagnose_junjul2026_losses.py per confronto
diretto: scomposizione per causa di uscita, ADX medio vincenti/perdenti,
distribuzione settimanale, per strumento — applicata sia al baseline
sia alla versione +GOLD, per vedere ESATTAMENTE dove GOLD avrebbe preso
il posto di un trade FTSE100 perdente (o di uno vincente, sarebbe
comunque un dato onesto da vedere).

Nessuna scrittura su D1. Solo lettura Dukascopy + stampa a log.
"""

from __future__ import annotations

import os
import pandas as pd
from datetime import datetime, timezone

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

SYMBOLS_2 = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
SYMBOLS_3 = dict(SYMBOLS_2, GOLD=INSTRUMENT_FX_METALS_XAU_USD)

WARMUP_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_START = datetime(2026, 6, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 7, 19, tzinfo=timezone.utc)

CAPITAL_V6 = 1400.0
CAPITAL_MR = 600.0


def fetch_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        WARMUP_START, TEST_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_from_test_start(df: pd.DataFrame, buffer_days: int = 2) -> pd.DataFrame:
    buf_start = pd.Timestamp(TEST_START) - pd.Timedelta(days=buffer_days)
    return df[df["timestamp"] >= buf_start].reset_index(drop=True)


def diagnose_trades(trades_df: pd.DataFrame, label: str, log) -> None:
    if trades_df.empty:
        log(f"  [{label}] nessun trade.")
        return
    n = len(trades_df)
    log(f"  [{label}] {n} trade totali, PnL={trades_df['pnl'].sum():+.2f} EUR")

    log(f"  [{label}] Causa di uscita:")
    for reason, group in trades_df.groupby("exit_reason"):
        wins = (group["pnl"] > 0).sum()
        log(f"      {reason:<12} n={len(group):>3}  vinti={wins:>3} ({wins/len(group)*100:.0f}%)  "
            f"PnL={group['pnl'].sum():+.2f} EUR")

    if "adx_at_entry" in trades_df.columns:
        wins_df = trades_df[trades_df["pnl"] > 0]
        losses_df = trades_df[trades_df["pnl"] <= 0]
        adx_w = wins_df["adx_at_entry"].mean() if not wins_df.empty else float("nan")
        adx_l = losses_df["adx_at_entry"].mean() if not losses_df.empty else float("nan")
        log(f"  [{label}] ADX medio — vincenti: {adx_w:.2f}  perdenti: {adx_l:.2f}")

    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["week"] = trades_df["entry_time"].dt.to_period("W").astype(str)
    log(f"  [{label}] Per settimana:")
    for week, group in trades_df.groupby("week"):
        wins = (group["pnl"] > 0).sum()
        log(f"      {week}: {len(group)} trade, {wins} vinti, PnL={group['pnl'].sum():+.2f} EUR")

    log(f"  [{label}] Per strumento:")
    for instrument, group in trades_df.groupby("instrument"):
        wins = (group["pnl"] > 0).sum()
        log(f"      {instrument}: {len(group)} trade, {wins} vinti ({wins/len(group)*100:.0f}%), "
            f"PnL={group['pnl'].sum():+.2f} EUR")
    log("")


def main():
    os.makedirs("results", exist_ok=True)
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Simulazione giugno-luglio 2026 CON GOLD — confronto diretto baseline vs +GOLD ===\n")

    log("Scarico storico DAX/FTSE100/GOLD...")
    raw = {name: fetch_full(const) for name, const in SYMBOLS_3.items()}
    log("Fatto.\n")

    instruments_2 = dict(eng.INSTRUMENTS)
    instruments_3 = instruments_with_gold()

    # ================================================================
    # V6
    # ================================================================
    log("=" * 70)
    log("V6 — baseline (DAX+FTSE100) vs +GOLD (DAX+FTSE100+GOLD)")
    log("=" * 70)

    v6_sig_3 = {name: slice_from_test_start(eng.generate_signals(raw[name], instruments_3[name]))
                for name in SYMBOLS_3}
    v6_sig_2 = {k: v for k, v in v6_sig_3.items() if k != "GOLD"}

    eng_v6_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
    trades_v6_base, _ = eng_v6_base.run(v6_sig_2)
    diagnose_trades(trades_v6_base, "V6 baseline", log)

    eng_v6_gold = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
    trades_v6_gold, _ = eng_v6_gold.run(v6_sig_3)
    diagnose_trades(trades_v6_gold, "V6 +GOLD", log)

    delta_v6 = trades_v6_gold['pnl'].sum() - trades_v6_base['pnl'].sum() if not trades_v6_gold.empty else -trades_v6_base['pnl'].sum()
    log(f">>> V6 Delta PnL (giugno-luglio 2026): {delta_v6:+.2f} EUR\n")

    # ================================================================
    # Mean-reversion
    # ================================================================
    log("=" * 70)
    log("Mean-reversion — baseline (DAX+FTSE100) vs +GOLD (DAX+FTSE100+GOLD)")
    log("=" * 70)

    mr_sig_3 = {name: slice_from_test_start(generate_mean_reversion_signals(raw[name], instruments_3[name], mode="rsi"))
                for name in SYMBOLS_3}
    mr_sig_2 = {k: v for k, v in mr_sig_3.items() if k != "GOLD"}

    eng_mr_base = BacktestEngineMeanReversion(capital0=CAPITAL_MR, instruments=instruments_2)
    trades_mr_base, _ = eng_mr_base.run(mr_sig_2)
    diagnose_trades(trades_mr_base, "MR baseline", log)

    eng_mr_gold = BacktestEngineMeanReversionGold(capital0=CAPITAL_MR, instruments=instruments_3)
    trades_mr_gold, _ = eng_mr_gold.run(mr_sig_3)
    diagnose_trades(trades_mr_gold, "MR +GOLD", log)

    delta_mr = trades_mr_gold['pnl'].sum() - trades_mr_base['pnl'].sum() if not trades_mr_gold.empty else -trades_mr_base['pnl'].sum()
    log(f">>> MR Delta PnL (giugno-luglio 2026): {delta_mr:+.2f} EUR\n")

    # ================================================================
    # Riepilogo
    # ================================================================
    log("=" * 70)
    log("RIEPILOGO")
    log("=" * 70)
    log(f"V6:  baseline PnL={trades_v6_base['pnl'].sum():+.2f}  "
        f"+GOLD PnL={(trades_v6_gold['pnl'].sum() if not trades_v6_gold.empty else 0.0):+.2f}  "
        f"delta={delta_v6:+.2f}")
    log(f"MR:  baseline PnL={trades_mr_base['pnl'].sum():+.2f}  "
        f"+GOLD PnL={(trades_mr_gold['pnl'].sum() if not trades_mr_gold.empty else 0.0):+.2f}  "
        f"delta={delta_mr:+.2f}")
    combined_base = trades_v6_base['pnl'].sum() + trades_mr_base['pnl'].sum()
    combined_gold = (trades_v6_gold['pnl'].sum() if not trades_v6_gold.empty else 0.0) + \
                    (trades_mr_gold['pnl'].sum() if not trades_mr_gold.empty else 0.0)
    log(f"Totale combinato corretto: baseline={combined_base:+.2f}  +GOLD={combined_gold:+.2f}  "
        f"delta={combined_gold-combined_base:+.2f}")

    with open("results/simulate_junjul2026_with_gold.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
