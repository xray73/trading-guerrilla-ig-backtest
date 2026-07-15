"""
extra_slot_15day_test.py — Analisi descrittiva (NON un test di
validazione) di quanti trade sarebbero scattati sugli slot extra
4°/5° nella stessa finestra di 15 giorni feriali usata da
refine_exit_15day_test.py (24/06-14/07/2026 circa, ricalcolata
dinamicamente rispetto a "ieri").

CONTESTO: il meccanismo slot extra (BacktestEngineExtendedOrders,
già in repo) è stato testato con disciplina train/test corretta +
bootstrap il 15/07/2026 e NON adottato — rapporto segnale/rumore sul
test (2026-ytd) = 0.012, indistinguibile dal caso (vedi
extended_orders_train_test.py e verdetto in results/). Questo script
NON riapre quella decisione: produce solo un numero descrittivo
("quanti trade extra sarebbero scattati in questi 15 giorni specifici
e come sarebbero finiti") come strumento di analisi collaterale al
monitoraggio pre-IG, nient'altro.

Meccanismo slot extra (invariato, da engine_extended_orders.py):
  - Slot 1-3: rischio standard, invariati.
  - Slot 4, 5: rischio = min(extra_slot_pct * PnL_netto_giornata_finora,
    rischio_standard_strumento). Se PnL netto giornata <= 0, slot
    NON si apre. Se il rischio modulato non copre la size minima,
    slot saltato (non forzato).
  - extra_slot_pct=1.0, max_new_orders_per_day=5 (parametri già
    fissati in chat, invariati).

Nessuna modifica al motore. Nessuna scrittura su D1. Solo CSV locale.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_extended_orders import BacktestEngineExtendedOrders

WARMUP_DAYS = 90
N_TRADING_DAYS = 15
EXTRA_SLOT_PCT = 1.0
CAPITAL0 = 2000.0
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
    day = end_exclusive
    counted = 0
    start = day
    while counted < n_days:
        start -= timedelta(days=1)
        if start.weekday() < 5:
            counted += 1
    return start, end_exclusive


def main():
    yesterday_end = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) \
        + timedelta(days=1)
    window_start, window_end = trading_days_window(N_TRADING_DAYS, yesterday_end)
    warmup_start = window_start - timedelta(days=WARMUP_DAYS)

    window_start_utc = pd.Timestamp(window_start, tz="UTC")
    window_end_utc = pd.Timestamp(window_end, tz="UTC")

    print(f"Finestra campione: {window_start.date()} -> {window_end.date()} ({N_TRADING_DAYS} giorni feriali)")

    full_data_30m = {}
    for name, const in SYMBOLS.items():
        print(f"Scarico {name} 30min ({warmup_start.date()} -> {window_end.date()})...")
        full_data_30m[name] = fetch_bars(const, warmup_start, window_end, dukascopy_python.INTERVAL_MIN_30)
        print(f"  {len(full_data_30m[name])} barre 30min")

    signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(full_data_30m[name], inst)

    # baseline: motore standard, max 3 ordini/giorno (invariato)
    print("\nEseguo motore baseline (max 3 ordini/giorno)...")
    engine_baseline = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_baseline, metrics_baseline = engine_baseline.run(signal_data)

    # esteso: stessi dati, stesso punto di partenza, slot 4-5 attivi
    print("Eseguo motore con slot extra (max 5 ordini/giorno, extra_slot_pct=1.0)...")
    p_extended = dataclasses.replace(eng.PARAMS, max_new_orders_per_day=5)
    engine_extended = BacktestEngineExtendedOrders(capital0=CAPITAL0, p=p_extended, extra_slot_pct=EXTRA_SLOT_PCT)
    trades_extended, metrics_extended = engine_extended.run(signal_data)

    trades_extended["entry_time"] = pd.to_datetime(trades_extended["entry_time"], utc=True)

    # identifica i trade extra (slot 4°/5°) tramite il log dedicato del motore
    extra_keys = set(engine_extended.extra_slot_log)  # {(instrument, entry_time), ...}
    trades_extended["is_extra_slot"] = trades_extended.apply(
        lambda r: (r["instrument"], r["entry_time"]) in extra_keys, axis=1
    )

    extra_trades = trades_extended[
        trades_extended["is_extra_slot"]
        & (trades_extended["entry_time"] >= window_start_utc)
        & (trades_extended["entry_time"] < window_end_utc)
    ].copy()

    baseline_window = trades_baseline[
        (pd.to_datetime(trades_baseline["entry_time"], utc=True) >= window_start_utc)
        & (pd.to_datetime(trades_baseline["entry_time"], utc=True) < window_end_utc)
    ].copy()

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — finestra {N_TRADING_DAYS} giorni feriali")
    print(f"{'='*70}")
    print(f"Trade baseline (slot 1-3) nella finestra: {len(baseline_window)}")
    print(f"Trade su slot extra (4°/5°) nella finestra: {len(extra_trades)}")
    print(f"  di cui saltati per PnL giornata <=0: {engine_extended.n_extra_slot_skipped_pnl}")
    print(f"  di cui saltati per size minima non coperta: {engine_extended.n_extra_slot_skipped_min_size}")

    if not extra_trades.empty:
        wins = (extra_trades["pnl"] > 0).sum()
        losses = (extra_trades["pnl"] <= 0).sum()
        print(f"\nEsito slot extra: {wins} vincenti / {losses} perdenti su {len(extra_trades)}")
        print(f"Win rate slot extra: {100*wins/len(extra_trades):.1f}%")
        print(f"PnL totale slot extra: {extra_trades['pnl'].sum():.2f}")
        print(f"PnL medio per trade extra: {extra_trades['pnl'].mean():.2f}")
        print("\nDettaglio:")
        for _, t in extra_trades.iterrows():
            esito = "WIN " if t["pnl"] > 0 else "LOSS"
            print(f"  {t['instrument']:8s} {t['direction']:5s} entry={t['entry_time']} "
                  f"exit_reason={t['exit_reason']:12s} pnl={t['pnl']:+8.2f}  [{esito}]")
    else:
        print("\nNessuno slot extra si sarebbe aperto in questa finestra "
              "(giornate senza 4°+ segnale, o PnL giornata già <=0 quando sarebbe scattato).")

    extra_trades.to_csv("extra_slot_15day_result.csv", index=False)
    print(f"\nCompletato. File: extra_slot_15day_result.csv")
    print("\nNota: risultato descrittivo su una finestra di 15 giorni, non un test di "
          "validazione. Il fronte 'slot extra' resta chiuso sulla base del test train/test "
          "+ bootstrap del 15/07/2026 (segnale/rumore = 0.012).")


if __name__ == "__main__":
    main()
