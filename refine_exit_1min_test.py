"""
refine_exit_1min_test.py — Verifica quanto la chiusura approssimata a
30min (come funziona oggi il motore, e come funzionerebbe la pipeline
live pre-IG) diverge dalla chiusura "vera" ricostruita a 1 minuto.

Procedura (proposta in chat 15/07/2026):
  1. Scarica barre 30min DAX+FTSE100 con warmup sufficiente, fino
     all'ultima giornata di trading completa disponibile
  2. Scarica barre 1min DAX+FTSE100 SOLO per quella giornata
  3. Fa girare il motore standard (BacktestEngineFloatingKillSwitch)
     sui dati a 30min, isola i trade aperti in quella giornata
  4. Per ciascun trade aperto quel giorno, ricostruisce l'uscita
     "vera" scansionando le barre da 1 minuto a partire dall'entry,
     e la confronta con l'uscita approssimata che il motore a 30min
     avrebbe deciso

Nessuna modifica al motore, nessuna nuova logica operativa — è solo
un'analisi di quanto conta l'approssimazione, prima di decidere se
vale la pena costruire il raffinamento nella pipeline pre-IG.

Non richiede D1 — analisi one-off, risultati stampati e salvati in
un CSV locale.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def fetch_bars(symbol_const, start: datetime, end: datetime, interval) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, interval, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def refine_exit(trade_row, bars_1m: pd.DataFrame, inst: eng.InstrumentConfig) -> dict:
    """Scansiona le barre 1min dall'entry in avanti, trova il primo
    minuto in cui stop o target vengono toccati davvero.

    NOTA: ClosedTrade non salva stop_loss/take_profit direttamente —
    li ricostruiamo con la STESSA formula usata da _open_position in
    engine.py (mai modificato, solo letto): stop_distance = ATR * moltiplicatore,
    target = stop_distance * rr_planned."""
    entry_time = trade_row["entry_time"]
    entry_price = trade_row["entry_price"]
    direction = trade_row["direction"]
    atr_at_entry = trade_row["atr_at_entry"]
    rr_planned = trade_row["rr_planned"]

    stop_distance = atr_at_entry * inst.atr_multiplier
    if direction == "long":
        stop = entry_price - stop_distance
        target = entry_price + stop_distance * rr_planned
    else:
        stop = entry_price + stop_distance
        target = entry_price - stop_distance * rr_planned

    window = bars_1m[bars_1m["timestamp"] > entry_time].reset_index(drop=True)
    for _, bar in window.iterrows():
        if direction == "long":
            if bar["low"] <= stop:
                return {"exit_time_1m": bar["timestamp"], "exit_price_1m": stop, "exit_reason_1m": "stop_loss"}
            if bar["high"] >= target:
                return {"exit_time_1m": bar["timestamp"], "exit_price_1m": target, "exit_reason_1m": "take_profit"}
        else:
            if bar["high"] >= stop:
                return {"exit_time_1m": bar["timestamp"], "exit_price_1m": stop, "exit_reason_1m": "stop_loss"}
            if bar["low"] <= target:
                return {"exit_time_1m": bar["timestamp"], "exit_price_1m": target, "exit_reason_1m": "take_profit"}
    return {"exit_time_1m": None, "exit_price_1m": None, "exit_reason_1m": "non_chiuso_entro_dati_1min"}


def main():
    # ultima giornata di trading completa: ieri (oggi potrebbe essere incompleto)
    target_day = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = target_day + timedelta(days=1)
    warmup_start = target_day - timedelta(days=WARMUP_DAYS)

    # versioni timezone-aware (UTC), usate SOLO per confrontare con
    # entry_time del DataFrame trade (già tz-aware dopo pd.to_datetime
    # con utc=True) — dukascopy_python.fetch continua a ricevere le
    # versioni naive sopra, coerente col resto degli script del progetto
    target_day_utc = pd.Timestamp(target_day, tz="UTC")
    day_end_utc = pd.Timestamp(day_end, tz="UTC")

    print(f"Giornata target: {target_day.date()}")

    full_data_30m = {}
    data_1m = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name} 30min ({warmup_start.date()} -> {day_end.date()})...")
        full_data_30m[name] = fetch_bars(const, warmup_start, day_end, dukascopy_python.INTERVAL_MIN_30)
        print(f"  {len(full_data_30m[name])} barre 30min")

        print(f"Scarico {name} 1min (solo {target_day.date()})...")
        data_1m[name] = fetch_bars(const, target_day, day_end, dukascopy_python.INTERVAL_MIN_1)
        print(f"  {len(data_1m[name])} barre 1min")

    # genera segnali sui dati 30min (stesso codice del backtest)
    signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(full_data_30m[name], inst)

    engine_ = BacktestEngineFloatingKillSwitch(capital0=2000.0)
    trades_df, metrics_df = engine_.run(signal_data)

    if trades_df.empty:
        print("\nNessun trade generato nell'intera finestra (warmup+giornata target). "
              "Prova un'altra giornata o verifica i dati.")
        return

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    day_trades = trades_df[
        (trades_df["entry_time"] >= target_day_utc) & (trades_df["entry_time"] < day_end_utc)
    ].copy()

    print(f"\nTrade aperti nella giornata target: {len(day_trades)}")
    if day_trades.empty:
        print("Nessun trade aperto in questa giornata specifica — riprova con un'altra data "
              "(il segnale non genera trigger tutti i giorni).")
        return

    rows = []
    for _, trade in day_trades.iterrows():
        instrument = trade["instrument"]
        inst = eng.INSTRUMENTS[instrument]
        refined = refine_exit(trade, data_1m[instrument], inst)

        exit_30m_time = pd.to_datetime(trade["exit_time"], utc=True) if pd.notna(trade["exit_time"]) else None
        exit_1m_time = refined["exit_time_1m"]

        delay_minutes = None
        if exit_30m_time is not None and exit_1m_time is not None:
            delay_minutes = (exit_30m_time - exit_1m_time).total_seconds() / 60

        rows.append({
            "instrument": instrument, "direction": trade["direction"],
            "entry_time": trade["entry_time"], "entry_price": trade["entry_price"],
            "exit_reason_30m": trade["exit_reason"], "exit_time_30m": exit_30m_time,
            "exit_price_30m": trade["exit_price"], "pnl_30m": trade["pnl"],
            "exit_reason_1m": refined["exit_reason_1m"], "exit_time_1m": exit_1m_time,
            "exit_price_1m": refined["exit_price_1m"],
            "ritardo_minuti_30m_vs_1m": delay_minutes,
            "stessa_causale": trade["exit_reason"] == refined["exit_reason_1m"],
        })
        print(f"  {instrument} {trade['direction']}: 30min={trade['exit_reason']} @ {exit_30m_time} | "
              f"1min={refined['exit_reason_1m']} @ {exit_1m_time} | "
              f"ritardo={delay_minutes:.1f}min" if delay_minutes is not None else
              f"  {instrument} {trade['direction']}: 30min={trade['exit_reason']} | 1min=dati insufficienti")

    result_df = pd.DataFrame(rows)
    result_df.to_csv("refine_exit_1min_result.csv", index=False)

    print(f"\n{'='*70}")
    print("RIEPILOGO")
    print(f"{'='*70}")
    print(f"Trade analizzati: {len(result_df)}")
    print(f"Causale di uscita coincidente (30min vs 1min): "
          f"{result_df['stessa_causale'].sum()}/{len(result_df)}")
    valid_delays = result_df["ritardo_minuti_30m_vs_1m"].dropna()
    if len(valid_delays) > 0:
        print(f"Ritardo medio 30min vs 1min: {valid_delays.mean():.1f} minuti")
        print(f"Ritardo massimo: {valid_delays.max():.1f} minuti")

    print(f"\nCompletato. File: refine_exit_1min_result.csv")


if __name__ == "__main__":
    main()
