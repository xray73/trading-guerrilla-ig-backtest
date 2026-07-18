"""
diagnose_junjul2026_losses.py — Diagnostica ad-hoc (18/07/2026):
scompone i trade del confronto V6 pura vs V6+MR (periodo 01/06/2026-
18/07/2026, vedi compare_v6_vs_v6mr_junjul2026.py) per capire PERCHE'
il win rate osservato (26-27%) sia stato così sotto il baseline
storico (38-40%) — non solo QUANTO.

Tre blocchi analizzati, tutti con la STESSA metodologia (stesso
periodo, stesso spread originale, stessi dati):
  1. V6 Scenario A (run continuo, capitale 2000 EUR) — il blocco
     principale, 46 trade attesi.
  2. V6 Scenario B (due blocchi mensili, capitale 1400 EUR) — stessi
     segnali di A ma motore chunked, ~47 trade attesi (vedi nota sulla
     semplificazione nota nello script di confronto). Confrontato con
     (1) come controllo di coerenza: se la scomposizione per causa di
     uscita è simile, il chunking mensile non ha distorto il quadro.
  3. Mean-reversion Scenario B — solo 1 trade, riportato individualmente
     (campione troppo piccolo per statistiche aggregate).

Per ciascun blocco V6, riporta (SOLO aggregati, mai singoli trade
grezzi in chat, coerente con le regole del progetto):
  - Scomposizione per exit_reason (stop_loss / take_profit /
    max_holding), conteggio e PnL per categoria — una quota alta di
    max_holding indicherebbe mercato laterale senza follow-through
    (nè stop nè target toccati), coerente con l'ipotesi ADX<20
    frequente già osservata nel controllo dei segnali mancati del
    16-17/07.
  - ADX medio all'apertura per trade vincenti vs perdenti — se i
    perdenti hanno sistematicamente ADX più basso all'ingresso,
    supporta l'ipotesi "falsi breakout in laterale" anche per i trade
    che V6 comunque genera (ADX>20 è già un filtro, ma il margine
    sopra soglia può ancora essere debole).
  - Distribuzione settimanale di trade e PnL, per capire se il
    problema è concentrato in poche settimane o diffuso uniformemente.

Nessuna scrittura su D1. Solo stampa a log + file risultati/ per l'artifact.
"""

from __future__ import annotations

import os
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100
from datetime import datetime, timezone

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_accantonamento import BacktestEngineAccantonamento
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
WARMUP_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_START = datetime(2026, 6, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 7, 19, tzinfo=timezone.utc)

CAPITAL0 = 2000.0
SPLIT_V6, SPLIT_MR = 0.70, 0.30
MR_MODE = "rsi"

MONTH_CHUNKS = [
    (datetime(2026, 6, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc)),
    (datetime(2026, 7, 1, tzinfo=timezone.utc), TEST_END),
]


def fetch_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        WARMUP_START, TEST_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_df(df: pd.DataFrame, start: datetime, end: datetime, buffer_days: int = 2) -> pd.DataFrame:
    buf_start = pd.Timestamp(start) - pd.Timedelta(days=buffer_days)
    return df[(df["timestamp"] >= buf_start) & (df["timestamp"] < pd.Timestamp(end))].reset_index(drop=True)


def diagnose_trades(trades_df: pd.DataFrame, label: str, log) -> None:
    if trades_df.empty:
        log(f"  [{label}] nessun trade, niente da diagnosticare.")
        return

    n = len(trades_df)
    log(f"  [{label}] {n} trade totali")

    # --- scomposizione per exit_reason ---
    log(f"  [{label}] Scomposizione per causa di uscita:")
    for reason, group in trades_df.groupby("exit_reason"):
        wins = (group["pnl"] > 0).sum()
        pnl_sum = group["pnl"].sum()
        log(f"      {reason:<12} n={len(group):>3}  vinti={wins:>3} ({wins/len(group)*100:.0f}%)  "
            f"PnL totale={pnl_sum:+.2f} EUR")

    # --- ADX medio: vincenti vs perdenti ---
    if "adx_at_entry" in trades_df.columns:
        wins_df = trades_df[trades_df["pnl"] > 0]
        losses_df = trades_df[trades_df["pnl"] <= 0]
        adx_w = wins_df["adx_at_entry"].mean() if not wins_df.empty else float("nan")
        adx_l = losses_df["adx_at_entry"].mean() if not losses_df.empty else float("nan")
        log(f"  [{label}] ADX medio all'ingresso — vincenti: {adx_w:.2f}  perdenti: {adx_l:.2f}  "
            f"(differenza: {adx_w - adx_l:+.2f})")

    # --- distribuzione settimanale ---
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["week"] = trades_df["entry_time"].dt.to_period("W").astype(str)
    log(f"  [{label}] Distribuzione settimanale:")
    for week, group in trades_df.groupby("week"):
        wins = (group["pnl"] > 0).sum()
        log(f"      {week}: {len(group)} trade, {wins} vinti, PnL={group['pnl'].sum():+.2f} EUR")

    # --- per strumento ---
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

    log(f"=== Diagnostica trade giugno-luglio 2026 ===\n")

    log("Scarico storico DAX/FTSE100...")
    hist_full = {}
    for name, const in SYMBOLS.items():
        hist_full[name] = fetch_full(const)
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(hist_full[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    mr_signals_full = {name: generate_mean_reversion_signals(hist_full[name], eng.INSTRUMENTS[name], mode=MR_MODE) for name in SYMBOLS}

    # ================================================================
    # BLOCCO 1 — V6 Scenario A (run continuo)
    # ================================================================
    log("=" * 70)
    log("BLOCCO 1 — V6 Scenario A (run continuo, capitale 2000 EUR)")
    log("=" * 70)
    v6_sliced_a = {name: slice_df(v6_signals_full[name], TEST_START, TEST_END) for name in SYMBOLS}
    engine_a = BacktestEngineAccantonamento(capital0=CAPITAL0, mode="gradini", check_frequency="mensile")
    trades_a, _ = engine_a.run(v6_sliced_a)
    diagnose_trades(trades_a, "V6-A", log)

    # ================================================================
    # BLOCCO 2 — V6 Scenario B (chunked mensile)
    # ================================================================
    log("=" * 70)
    log("BLOCCO 2 — V6 Scenario B (chunked mensile, capitale 1400 EUR)")
    log("=" * 70)
    all_trades_v6_b = []
    cap_v6 = CAPITAL0 * SPLIT_V6
    for chunk_start, chunk_end in MONTH_CHUNKS:
        v6_sliced = {name: slice_df(v6_signals_full[name], chunk_start, chunk_end) for name in SYMBOLS}
        eng_v6 = BacktestEngineFloatingKillSwitch(capital0=cap_v6)
        trades_v6, _ = eng_v6.run(v6_sliced)
        if not trades_v6.empty:
            all_trades_v6_b.append(trades_v6)
        cap_v6 = eng_v6.capital
    trades_b_v6 = pd.concat(all_trades_v6_b, ignore_index=True) if all_trades_v6_b else pd.DataFrame()
    diagnose_trades(trades_b_v6, "V6-B", log)

    # ================================================================
    # BLOCCO 3 — Mean-reversion Scenario B
    # ================================================================
    log("=" * 70)
    log("BLOCCO 3 — Mean-reversion Scenario B (capitale 600 EUR)")
    log("=" * 70)
    all_trades_mr_b = []
    cap_mr = CAPITAL0 * SPLIT_MR
    for chunk_start, chunk_end in MONTH_CHUNKS:
        mr_sliced = {name: slice_df(mr_signals_full[name], chunk_start, chunk_end) for name in SYMBOLS}
        eng_mr = BacktestEngineMeanReversion(capital0=cap_mr)
        trades_mr, _ = eng_mr.run(mr_sliced)
        if not trades_mr.empty:
            all_trades_mr_b.append(trades_mr)
        cap_mr = eng_mr.capital
    trades_b_mr = pd.concat(all_trades_mr_b, ignore_index=True) if all_trades_mr_b else pd.DataFrame()
    if trades_b_mr.empty:
        log("  Nessun trade mean-reversion nel periodo.")
    else:
        for _, t in trades_b_mr.iterrows():
            log(f"  Trade unico: {t['instrument']} {t['direction']} entry={t['entry_time']} "
                f"exit={t['exit_time']} ({t['exit_reason']}) pnl={t['pnl']:+.2f} EUR "
                f"ADX_entry={t.get('adx_at_entry', 'N/A')}")
        log("  (campione troppo piccolo, n=1, per qualunque statistica aggregata)")

    with open("results/diagnose_junjul2026_losses.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
