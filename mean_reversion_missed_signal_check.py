"""
mean_reversion_missed_signal_check.py — Diagnostica ad-hoc (18/07/2026):
verifica se l'assenza di segnali il 16-17/07/2026 su DAX/FTSE100 sia
spiegata dal fatto che la mean-reversion (ramo esplorativo, NON ancora
integrata in live_execute.py) sia rimasta silenziosa nello stesso periodo,
o se avrebbe invece generato segnali/trade.

Metodologia:
  - Scarica storico 30min DAX/FTSE100 via Dukascopy con lo stesso warmup
    di live_execute.py (90gg), necessario per EMA200/ADX/Bollinger/RSI.
  - Calcola sia la variante Bollinger che RSI (mean_reversion_signals.py,
    invariato — nessuna modifica al modulo).
  - Simula con BacktestEngine standard (engine.py, invariato), capitale
    pieno 2000 EUR, SOLO sui segnali mean-reversion, finestra di
    simulazione ristretta a pochi giorni intorno al periodo in esame
    (per evitare che PnL di mesi di storico irrilevante influenzi il
    sizing — qui interessa solo "cosa sarebbe successo quei 2 giorni").
  - V6 non è ricalcolata qui: già confermato 0 segnali/0 ordini via
    query diretta su live_daily_state (orders_today=0 sia 16 che 17/07).

Nessuna scrittura su D1. Nessun impatto su live_execute.py o sul motore.
Solo diagnostica, stampata nel log della run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from mean_reversion_signals import generate_mean_reversion_signals

WARMUP_DAYS = 90
CHECK_LABEL_START = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)
CHECK_LABEL_END = datetime(2026, 7, 18, 0, 0, tzinfo=timezone.utc)
SIM_WINDOW_START = CHECK_LABEL_START - timedelta(days=2)  # margine per continuita' segnale prev-bar

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
CAPITAL0 = 2000.0


def fetch_historical(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    now = datetime.now(timezone.utc)
    print(f"=== mean_reversion_missed_signal_check.py — {now.isoformat()} ===")
    print(f"Periodo esaminato: {CHECK_LABEL_START} -> {CHECK_LABEL_END}\n")

    for name, const in SYMBOLS.items():
        inst = eng.INSTRUMENTS[name]
        warmup_start = SIM_WINDOW_START - timedelta(days=WARMUP_DAYS)
        hist = fetch_historical(const, warmup_start, now)

        window_check = hist[(hist["timestamp"] >= pd.Timestamp(CHECK_LABEL_START)) &
                             (hist["timestamp"] < pd.Timestamp(CHECK_LABEL_END))]
        print(f"--- {name} ---")
        if window_check.empty:
            print("  Nessuna barra scaricata nel periodo 16-17/07 (mercato chiuso o gap dati). Salto.\n")
            continue

        for mode in ("bollinger", "rsi"):
            sig_df_full = generate_mean_reversion_signals(hist, inst, mode=mode)

            win_sig = sig_df_full[(sig_df_full["timestamp"] >= pd.Timestamp(CHECK_LABEL_START)) &
                                   (sig_df_full["timestamp"] < pd.Timestamp(CHECK_LABEL_END))]
            adx_min, adx_max = win_sig["adx"].min(), win_sig["adx"].max()
            n_regime_ok = int((win_sig["adx"] < 20.0).sum())
            print(f"  [{mode}] ADX nel periodo: min={adx_min:.1f} max={adx_max:.1f} "
                  f"(barre con ADX<20: {n_regime_ok}/{len(win_sig)})")

            signals_found = win_sig[win_sig["signal"].isin(["long", "short"])]
            if signals_found.empty:
                print(f"  [{mode}] Nessun segnale mean-reversion nel periodo.\n")
                continue

            for _, row in signals_found.iterrows():
                extra = f" RSI={row['rsi']:.1f}" if "rsi" in row and pd.notna(row.get("rsi")) else ""
                print(f"  [{mode}] SEGNALE {row['signal'].upper()} alle {row['timestamp']} "
                      f"(ADX={row['adx']:.1f}, ATR={row['atr']:.2f}{extra})")

            # simulazione isolata: finestra ristretta, capitale fresco 2000 EUR
            sig_df_sim = sig_df_full[sig_df_full["timestamp"] >= pd.Timestamp(SIM_WINDOW_START)].reset_index(drop=True)
            engine_sim = eng.BacktestEngine(capital0=CAPITAL0)
            trades_df, _ = engine_sim.run({name: sig_df_sim})

            if trades_df.empty:
                print(f"  [{mode}] Segnale rilevato ma nessun trade aperto dal motore nella finestra "
                      f"di simulazione (possibile ATR/ADX NaN sulla barra di ingresso).\n")
                continue

            window_trades = trades_df[
                (pd.to_datetime(trades_df["entry_time"]) >= pd.Timestamp(CHECK_LABEL_START)) &
                (pd.to_datetime(trades_df["entry_time"]) < pd.Timestamp(CHECK_LABEL_END))
            ]
            if window_trades.empty:
                print(f"  [{mode}] Segnali rilevati ma nessun trade con ingresso nel periodo "
                      f"(l'ingresso avviene alla barra N+1 — verificare bordo finestra).\n")
                continue

            for _, t in window_trades.iterrows():
                print(f"  [{mode}] TRADE: {t['instrument']} {t['direction']} "
                      f"entry={t['entry_time']}@{t['entry_price']:.1f} "
                      f"exit={t['exit_time']}@{t['exit_price']:.1f} "
                      f"({t['exit_reason']}) pnl={t['pnl']:+.2f} EUR "
                      f"(size={t['size']:.2f}, R={t['r_multiple']:+.2f})")
            print()

    print("=== Completato. ===")


if __name__ == "__main__":
    main()
