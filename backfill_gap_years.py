"""
backfill_gap_years.py — Riempie gli anni mancanti in ohlc_prices
(2017-2019, 2021-2022, mai caricati finora — il progetto ha sempre
usato solo le finestre dei 5 periodi ufficiali, vedi 00_CURRENT_STATE.md
sez. "Convenzione dati OHLC"). Decisione presa il 22/07/2026: la
tabella diagnostica adx_diagnostic_raw ha già confermato che Dukascopy
HA dati continui per DAX/FTSE100 su questi anni (nessun buco alla
fonte) — questo script li carica anche in ohlc_prices, il dataset
ufficiale, in modo puramente ADDITIVO: nessun backtest già validato è
interessato, dato che tutti filtrano sempre per range di date esplicito
dei 5 periodi ufficiali.

DIFFERENZA CHIAVE rispetto a load_ohlc_generic.py: quello script fa
DELETE FROM ohlc_prices WHERE symbol=X prima di reinserire TUTTI i 5
periodi (reload completo). Questo script invece NON tocca le righe
esistenti — cancella solo l'intervallo di date specifico che sta per
reinserire (idempotente su riesecuzione, ma non cancella mai gli altri
anni già presenti per lo stesso simbolo).

LIMITE DI SICUREZZA: D1 free tier consente 100.000 righe scritte/giorno
(fonte: Cloudflare D1 FAQ, verificato 22/07/2026). Su richiesta
esplicita, questo script si ferma con errore PRIMA di generare il file
SQL se il conteggio totale supera MAX_ROWS_PER_RUN=50.000 — lascia
margine per cron live_execute (ogni 30min) e altre attività giornaliere
che scrivono su D1 lo stesso giorno.

AGGIORNAMENTO 23/07/2026: aggiunto EURUSD a SYMBOL_MAP (costante
INSTRUMENT_FX_MAJORS_EUR_USD, confermata via discover_dukascopy_eurusd.py
il 23/07/2026) — serve a colmare i 5 anni mancanti (2017-2019, 2021-2022)
dal backfill iniziale, che aveva coperto solo i 5 periodi ufficiali
(81.442 righe, verificato in D1 il 23/07/2026).

Uso:
  python backfill_gap_years.py DAX 2017,2018,2019
  python backfill_gap_years.py GOLD 2021,2022
  python backfill_gap_years.py EURUSD 2017,2018,2019
  python backfill_gap_years.py EURUSD 2021,2022
"""

from __future__ import annotations

import sys
from datetime import datetime

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_FX_METALS_XAU_USD, INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_EUR_USD,
)

SYMBOL_MAP = {
    "DAX": INSTRUMENT_IDX_EUROPE_E_DAAX,
    "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    "GOLD": INSTRUMENT_FX_METALS_XAU_USD,
    "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
}

MAX_ROWS_PER_RUN = 50_000
CHUNK_SIZE = 500


def fetch_year(instrument_const, year: int):
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    df = dukascopy_python.fetch(
        instrument_const,
        dukascopy_python.INTERVAL_MIN_30,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    return df.reset_index(), start, end


def main():
    if len(sys.argv) < 3:
        print("Uso: python backfill_gap_years.py SIMBOLO ANNO1,ANNO2,...")
        print(f"Simboli disponibili: {', '.join(SYMBOL_MAP)}")
        sys.exit(1)

    symbol = sys.argv[1].strip().upper()
    if symbol not in SYMBOL_MAP:
        print(f"ERRORE: simbolo '{symbol}' non riconosciuto. Disponibili: {', '.join(SYMBOL_MAP)}")
        sys.exit(1)

    try:
        years = sorted(int(y.strip()) for y in sys.argv[2].split(",") if y.strip())
    except ValueError:
        print("ERRORE: anni devono essere interi separati da virgola, es. 2017,2018,2019")
        sys.exit(1)

    print(f"=== Backfill {symbol}, anni: {years} ===")

    all_rows = []
    per_year_counts = {}
    overall_start = None
    overall_end = None

    for year in years:
        print(f"  Scarico {year}...")
        df, y_start, y_end = fetch_year(SYMBOL_MAP[symbol], year)
        n = len(df)
        per_year_counts[year] = n
        print(f"    {n} barre scaricate per {year}")
        if n == 0:
            print(f"    ATTENZIONE: zero barre per {year} — possibile buco alla fonte, verificare manualmente.")
            continue

        overall_start = y_start if overall_start is None else min(overall_start, y_start)
        overall_end = y_end if overall_end is None else max(overall_end, y_end)

        ts_col = df.columns[0]
        for _, row in df.iterrows():
            ts_str = row[ts_col].strftime("%Y-%m-%d %H:%M:%S+00:00")
            vol = row["volume"] if "volume" in df.columns and row["volume"] == row["volume"] else 0
            all_rows.append((
                symbol, ts_str, "30m",
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(vol),
            ))

    total = len(all_rows)
    print(f"\n  Totale righe raccolte: {total} (limite per run: {MAX_ROWS_PER_RUN})")
    print(f"  Dettaglio per anno: {per_year_counts}")

    if total == 0:
        print("  ERRORE: nessuna barra scaricata per nessun anno richiesto. Interrompo, nessun file generato.")
        sys.exit(1)

    if total > MAX_ROWS_PER_RUN:
        print(f"\n  ERRORE: {total} righe superano il limite di sicurezza {MAX_ROWS_PER_RUN}/run "
              f"(margine per cron live_execute + altre attività giornaliere su D1).")
        print("  Nessun file SQL generato. Dividi gli anni richiesti in gruppi più piccoli e rilancia.")
        sys.exit(1)

    filename = f"backfill_{symbol.lower()}_{years[0]}_{years[-1]}.sql"
    with open(filename, "w") as f:
        f.write(
            f"DELETE FROM ohlc_prices WHERE symbol='{symbol}' "
            f"AND timestamp >= '{overall_start.strftime('%Y-%m-%d %H:%M:%S+00:00')}' "
            f"AND timestamp < '{overall_end.strftime('%Y-%m-%d %H:%M:%S+00:00')}';\n\n"
        )
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

    print(f"\n  File {filename} generato con {total} righe (sotto il limite, OK per il caricamento).")

    with open("generated_sql_files.txt", "w") as f:
        f.write(filename)

    print(f"\n=== Completato. File pronto: {filename} ===")


if __name__ == "__main__":
    main()
