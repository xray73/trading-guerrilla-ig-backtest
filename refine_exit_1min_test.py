"""
refine_exit_15day_test.py — Estensione a campione multi-giornata di
refine_exit_1min_test.py.

Confronta la chiusura approssimata a 30min (motore standard) con la
ricostruzione a 1 minuto, su un campione di N_TRADING_DAYS giornate di
trading consecutive (invece della singola giornata del test originale),
per avere statistiche aggregate minimamente significative.

Convenzione timestamp confermata (15/07/2026): le barre Dukascopy sono
etichettate con l'INIZIO del periodo. Il "ritardo" misurato tra
exit_time_30m e exit_time_1m è quindi una quantità strutturale attesa
(quanto in profondità nella barra da 30min è scattato realmente lo
stop/target), non un artefatto da correggere.

Nessuna modifica al motore, nessuna nuova logica operativa — analisi
one-off, risultati stampati e salvati in CSV locale.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
N_TRADING_DAYS = 15  # numero di giornate di trading da includere nel campione
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def fetch_bars(symbol_const, start: datetime, end: datetime, interval) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, interval, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def trading_days_window(n_days: int, end_exclusive: datetime) -> tuple[datetime, datetime]:
    """Ritorna (start, end) coprendo n_days giorni FERIALI (lun-ven)
    che terminano appena prima di end_exclusive. Approssimazione:
    non esclude festivi di borsa, solo weekend — sufficiente per
    dimensionare la finestra di fetch."""
    day = end_exclusive
    counted = 0
    start = day
    while counted < n_days:
        start -= timedelta(days=1)
        if start.weekday() < 5:  # 0=lun ... 4=ven
            counted += 1
    return start, end_exclusive


def refine_exit(trade_row, bars_1m: pd.DataFrame, inst: eng.InstrumentConfig) -> dict:
    """Scansiona le barre 1min dall'entry in avanti, trova il primo
    minuto in cui stop o target vengono toccati davvero. Stessa
    formula di _open_position in engine.py (mai modificato, solo letto)."""
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
    # finestra campione: N_TRADING_DAYS giorni feriali fino a ieri (oggi potrebbe essere incompleto)
    yesterday_end = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) \
        + timedelta(days=1)
    window_start, window_end = trading_days_window(N_TRADING_DAYS, yesterday_end)
    warmup_start = window_start - timedelta(days=WARMUP_DAYS)

    window_start_utc = pd.Timestamp(window_start, tz="UTC")
    window_end_utc = pd.Timestamp(window_end, tz="UTC")

    print(f"Finestra campione: {window_start.date()} -> {window_end.date()} "
          f"({N_TRADING_DAYS} giorni feriali)")

    full_data_30m = {}
    data_1m = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name} 30min ({warmup_start.date()} -> {window_end.date()})...")
        full_data_30m[name] = fetch_bars(const, warmup_start, window_end, dukascopy_python.INTERVAL_MIN_30)
        print(f"  {len(full_data_30m[name])} barre 30min")

        print(f"Scarico {name} 1min ({window_start.date()} -> {window_end.date()})...")
        data_1m[name] = fetch_bars(const, window_start, window_end, dukascopy_python.INTERVAL_MIN_1)
        print(f"  {len(data_1m[name])} barre 1min")

    signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(full_data_30m[name], inst)

    engine_ = BacktestEngineFloatingKillSwitch(capital0=2000.0)
    trades_df, metrics_df = engine_.run(signal_data)

    if trades_df.empty:
        print("\nNessun trade generato nella finestra warmup+campione.")
        return

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    window_trades = trades_df[
        (trades_df["entry_time"] >= window_start_utc) & (trades_df["entry_time"] < window_end_utc)
    ].copy()

    print(f"\nTrade aperti nella finestra campione: {len(window_trades)}")
    if window_trades.empty:
        print("Nessun trade nel campione — prova ad allargare N_TRADING_DAYS.")
        return

    rows = []
    for _, trade in window_trades.iterrows():
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

    result_df = pd.DataFrame(rows)
    result_df.to_csv("refine_exit_15day_result.csv", index=False)

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — campione {N_TRADING_DAYS} giorni feriali")
    print(f"{'='*70}")
    print(f"Trade analizzati: {len(result_df)}")
    coincident = result_df["stessa_causale"].sum()
    print(f"Causale di uscita coincidente (30min vs 1min): {coincident}/{len(result_df)} "
          f"({100*coincident/len(result_df):.1f}%)")

    valid_delays = result_df["ritardo_minuti_30m_vs_1m"].dropna()
    if len(valid_delays) > 0:
        print(f"Ritardo medio 30min vs 1min: {valid_delays.mean():.1f} minuti")
        print(f"Ritardo mediano: {valid_delays.median():.1f} minuti")
        print(f"Ritardo massimo: {valid_delays.max():.1f} minuti")
        print(f"Ritardo minimo: {valid_delays.min():.1f} minuti")

    non_chiusi = (result_df["exit_reason_1m"] == "non_chiuso_entro_dati_1min").sum()
    if non_chiusi > 0:
        print(f"\nATTENZIONE: {non_chiusi} trade non chiusi entro i dati 1min disponibili "
              f"(probabile uscita oltre la fine della finestra campione).")

    print(f"\nCompletato. File: refine_exit_15day_result.csv")


if __name__ == "__main__":
    main()
