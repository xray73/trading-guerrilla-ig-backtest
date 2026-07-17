"""
yahoo_dukascopy_compare.py — Secondo controllo indipendente sui dati
(17/07/2026): rigira Variante 6 (motore INVARIATO, engine.py non
toccato) su dati Yahoo Finance invece di Dukascopy, stessa finestra
temporale recente, per vedere se i risultati sono comparabili — se
sì, riduce il sospetto che Dukascopy stia "gonfiando" l'edge in modo
sistematico.

Limite di yfinance: barre a 30min disponibili solo per gli ultimi
~60 giorni di calendario (limite della piattaforma, non nostro).
Uso i primi ~30 giorni come warmup "naturale" degli indicatori
(EMA200/ADX hanno bisogno di storico) e confronto solo sugli ultimi
~30 giorni, dove gli indicatori sono più affidabili su entrambe le
fonti.

Ticker Yahoo: ^GDAXI (DAX), ^FTSE (FTSE100) — standard, ben noti.

Confronto su due livelli:
  1. Metriche aggregate (n trade, win rate, PF, PnL) — motore e
     parametri IDENTICI (BacktestEngineFloatingKillSwitch, invariato),
     unica differenza la fonte dati.
  2. Date dei segnali — le stesse giornate di breakout dovrebbero
     comparire su entrambe le fonti, indipendentemente da piccole
     differenze di prezzo/timestamp tra vendor.

Richiede yfinance (non ancora nelle dipendenze standard del progetto,
installata solo per questo script).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
COMPARE_DAYS = 30       # finestra di confronto vera e propria
TOTAL_FETCH_DAYS = 59   # tetto yfinance ~60gg per barre 30min, margine di sicurezza

YAHOO_TICKERS = {"DAX": "^GDAXI", "FTSE100": "^FTSE"}
DUKASCOPY_SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}


def fetch_yahoo_30m(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{TOTAL_FETCH_DAYS}d", interval="30m", progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    # yfinance a volte ritorna MultiIndex colonne (ticker, campo) — appiattisco se serve
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={ts_col: "timestamp", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp").reset_index(drop=True)


def fetch_dukascopy_30m(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "pnl_total": 0.0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    return {"n_trades": n, "win_rate_pct": 100 * len(wins) / n,
            "profit_factor": pf, "pnl_total": trades_df["pnl"].sum()}


def main():
    end = datetime.now()
    compare_start = end - timedelta(days=COMPARE_DAYS)
    total_start = end - timedelta(days=TOTAL_FETCH_DAYS)
    compare_start_ts = pd.Timestamp(compare_start, tz="UTC")

    print(f"=== Confronto Yahoo Finance vs Dukascopy — Variante 6 (motore invariato) ===")
    print(f"Finestra di confronto: {compare_start.date()} -> {end.date()} ({COMPARE_DAYS} giorni)")
    print(f"(primi {TOTAL_FETCH_DAYS - COMPARE_DAYS} giorni usati solo come warmup indicatori)\n")

    all_summary = []
    signal_dates = {"yahoo": {}, "dukascopy": {}}

    for name in ("DAX", "FTSE100"):
        inst = eng.INSTRUMENTS[name]

        print(f"--- {name} ---")
        print("Scarico Yahoo Finance...")
        yahoo_raw = fetch_yahoo_30m(YAHOO_TICKERS[name])
        print(f"  {len(yahoo_raw)} barre Yahoo")

        print("Scarico Dukascopy (stessa finestra)...")
        duka_raw = fetch_dukascopy_30m(DUKASCOPY_SYMBOLS[name], total_start, end)
        print(f"  {len(duka_raw)} barre Dukascopy")

        if yahoo_raw.empty:
            print(f"  ATTENZIONE: nessun dato Yahoo per {name}, salto questo strumento.\n")
            continue

        yahoo_signals = eng.generate_signals(yahoo_raw, inst)
        duka_signals = eng.generate_signals(duka_raw, inst)

        yahoo_window = yahoo_signals[yahoo_signals["timestamp"] >= compare_start_ts].reset_index(drop=True)
        duka_window = duka_signals[duka_signals["timestamp"] >= compare_start_ts].reset_index(drop=True)

        engine_yahoo = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_yahoo, _ = engine_yahoo.run({name: yahoo_window})

        engine_duka = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_duka, _ = engine_duka.run({name: duka_window})

        m_yahoo = metrics_summary(trades_yahoo)
        m_duka = metrics_summary(trades_duka)

        print(f"  Yahoo:     n={m_yahoo['n_trades']} WR={m_yahoo['win_rate_pct']:.1f}% "
              f"PF={m_yahoo['profit_factor']:.2f} PnL={m_yahoo['pnl_total']:+.2f}"
              if m_yahoo['n_trades'] > 0 else "  Yahoo: nessun trade")
        print(f"  Dukascopy: n={m_duka['n_trades']} WR={m_duka['win_rate_pct']:.1f}% "
              f"PF={m_duka['profit_factor']:.2f} PnL={m_duka['pnl_total']:+.2f}\n"
              if m_duka['n_trades'] > 0 else "  Dukascopy: nessun trade\n")

        all_summary.append({"strumento": name, "fonte": "yahoo", **m_yahoo})
        all_summary.append({"strumento": name, "fonte": "dukascopy", **m_duka})

        # confronto date dei segnali (giorno di ENTRY, non ora esatta)
        yahoo_sig_dates = set(pd.to_datetime(trades_yahoo["entry_time"]).dt.date) if not trades_yahoo.empty else set()
        duka_sig_dates = set(pd.to_datetime(trades_duka["entry_time"]).dt.date) if not trades_duka.empty else set()
        signal_dates["yahoo"][name] = yahoo_sig_dates
        signal_dates["dukascopy"][name] = duka_sig_dates

        common = yahoo_sig_dates & duka_sig_dates
        only_yahoo = yahoo_sig_dates - duka_sig_dates
        only_duka = duka_sig_dates - yahoo_sig_dates
        print(f"  Giorni con trade in comune: {len(common)}")
        print(f"  Solo su Yahoo: {sorted(only_yahoo)}")
        print(f"  Solo su Dukascopy: {sorted(only_duka)}\n")

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv("yahoo_dukascopy_compare_results.csv", index=False)
    print("Completato. File: yahoo_dukascopy_compare_results.csv")


if __name__ == "__main__":
    main()
