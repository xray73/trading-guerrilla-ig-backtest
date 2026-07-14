"""
load_italy40_data.py — Scarica Italy 40 (MIB, INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR)
da Dukascopy per i 5 periodi standard del progetto, genera un file SQL
di caricamento per D1 (stessa tabella ohlc_prices, stesse convenzioni
di DAX/FTSE100/GOLD/US500 — timeframe="30m", source="dukascopy").

Costante strumento confermata tramite discover_dukascopy_instrument.py
(non indovinata): INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR = 'ITA.IDX/EUR'.

Idempotente: il file SQL generato cancella prima ogni riga con
symbol='ITALY40' esistente, poi ricarica — sicuro da rilanciare senza
duplicare dati se qualcosa va storto la prima volta.

Il file SQL va eseguito con wrangler (non con REST API diretta) per lo
stesso motivo già documentato nel progetto: bulk insert di migliaia di
righe è impraticabile via singole chiamate REST — Regole_Backtest_MonteCarlo.md.
"""

from __future__ import annotations

from datetime import datetime

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR

SYMBOL = "ITALY40"
CHUNK_SIZE = 500  # righe per singola istruzione INSERT

PERIODS = {
    "2015-2016": (datetime(2015, 1, 1), datetime(2017, 1, 1)),
    "2020-covid": (datetime(2020, 1, 1), datetime(2021, 1, 1)),
    "2023": (datetime(2023, 1, 1), datetime(2024, 1, 1)),
    "2024-2025": (datetime(2024, 1, 1), datetime(2026, 1, 1)),
    "2026-ytd": (datetime(2026, 1, 1), datetime(2026, 7, 14)),
}


def fetch_period(start: datetime, end: datetime):
    df = dukascopy_python.fetch(
        INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR,
        dukascopy_python.INTERVAL_MIN_30,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    return df.reset_index()


def main():
    all_rows = []
    total_expected = 0

    for label, (start, end) in PERIODS.items():
        print(f"Scarico {label}: {start.date()} -> {end.date()}...")
        df = fetch_period(start, end)
        n = len(df)
        total_expected += n
        print(f"  {n} barre scaricate")

        if n == 0:
            print(f"  ATTENZIONE: zero barre per {label} — verificare manualmente.")
            continue

        ts_col = df.columns[0]  # la colonna indice resettata (timestamp)
        for _, row in df.iterrows():
            ts_str = row[ts_col].strftime("%Y-%m-%d %H:%M:%S+00:00")
            vol = row["volume"] if "volume" in df.columns and row["volume"] == row["volume"] else 0
            all_rows.append((
                SYMBOL, ts_str, "30m",
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(vol),
            ))

    print(f"\nTotale barre raccolte: {len(all_rows)} (attese: {total_expected})")

    if not all_rows:
        print("ERRORE: nessuna barra scaricata in nessun periodo. Non genero il file SQL.")
        return

    with open("insert_italy40.sql", "w") as f:
        # idempotente: rimuove eventuali dati precedenti dello stesso symbol
        # prima di ricaricare, per sicurezza in caso di ri-esecuzione
        f.write(f"DELETE FROM ohlc_prices WHERE symbol='{SYMBOL}';\n\n")

        for i in range(0, len(all_rows), CHUNK_SIZE):
            chunk = all_rows[i:i + CHUNK_SIZE]
            values = ",\n".join(
                f"('{s}', '{t}', '{tf}', {o}, {h}, {l}, {c}, {v}, 'dukascopy')"
                for (s, t, tf, o, h, l, c, v) in chunk
            )
            f.write(
                "INSERT INTO ohlc_prices "
                "(symbol, timestamp, timeframe, open, high, low, close, volume, source) "
                f"VALUES\n{values};\n\n"
            )

    print(f"File insert_italy40.sql generato con {len(all_rows)} righe "
          f"({(len(all_rows) + CHUNK_SIZE - 1) // CHUNK_SIZE} istruzioni INSERT).")


if __name__ == "__main__":
    main()
