"""
load_italy40_gap_fix.py — Ricarica SOLO la finestra mancante scoperta in
D1 per ITALY40 (aprile 2015 - giugno 2016, 15 mesi consecutivi a zero
barre — verificato, non un limite di copertura Dukascopy, un buco di
caricamento). Non tocca il resto dei dati già presenti e corretti
(verificato: 2020-covid, 2023, 2024-25, 2026-ytd tutti completi, densità
26 barre/giorno costante — struttura oraria di ITALY40, non un bug).

Idempotente sulla SOLA finestra del gap: cancella eventuali righe già
presenti in quell'intervallo specifico prima di ricaricare (mai l'intero
symbol, a differenza dello script di caricamento iniziale) — sicuro da
rilanciare senza toccare il resto dei dati validi.
"""

from __future__ import annotations

from datetime import datetime

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR

SYMBOL = "ITALY40"
CHUNK_SIZE = 500

# piccolo margine di un giorno oltre il gap noto (2015-04-01 -> 2016-07-01),
# per sicurezza sui bordi, senza sovrapporsi ai mesi gia' confermati completi
GAP_START = datetime(2015, 3, 31)
GAP_END = datetime(2016, 7, 1)


def main():
    print(f"Scarico finestra mancante: {GAP_START.date()} -> {GAP_END.date()}...")
    df = dukascopy_python.fetch(
        INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR,
        dukascopy_python.INTERVAL_MIN_30,
        dukascopy_python.OFFER_SIDE_BID,
        GAP_START,
        GAP_END,
    )
    df = df.reset_index()
    n = len(df)
    print(f"  {n} barre scaricate")

    if n == 0:
        print("ERRORE: zero barre scaricate per la finestra del gap — "
              "verificare manualmente prima di procedere. Nessun file SQL generato.")
        return

    ts_col = df.columns[0]
    rows = []
    for _, row in df.iterrows():
        ts_str = row[ts_col].strftime("%Y-%m-%d %H:%M:%S+00:00")
        vol = row["volume"] if "volume" in df.columns and row["volume"] == row["volume"] else 0
        rows.append((
            SYMBOL, ts_str, "30m",
            float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]), float(vol),
        ))

    gap_start_str = GAP_START.strftime("%Y-%m-%d %H:%M:%S+00:00")
    gap_end_str = GAP_END.strftime("%Y-%m-%d %H:%M:%S+00:00")

    with open("insert_italy40_gapfix.sql", "w") as f:
        # idempotente SOLO sulla finestra del gap, non tocca il resto del symbol
        f.write(
            f"DELETE FROM ohlc_prices WHERE symbol='{SYMBOL}' "
            f"AND timestamp >= '{gap_start_str}' AND timestamp < '{gap_end_str}';\n\n"
        )
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i:i + CHUNK_SIZE]
            values = ",\n".join(
                f"('{s}', '{t}', '{tf}', {o}, {h}, {l}, {c}, {v}, 'dukascopy')"
                for (s, t, tf, o, h, l, c, v) in chunk
            )
            f.write(
                "INSERT INTO ohlc_prices "
                "(symbol, timestamp, timeframe, open, high, low, close, volume, source) "
                f"VALUES\n{values};\n\n"
            )

    print(f"File insert_italy40_gapfix.sql generato con {n} righe "
          f"({(n + CHUNK_SIZE - 1) // CHUNK_SIZE} istruzioni INSERT).")


if __name__ == "__main__":
    main()
