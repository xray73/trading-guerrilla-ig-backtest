"""
load_ohlc_generic.py — Versione generica di load_italy40_data.py /
load_smi_data.py: accetta una lista di simboli da riga di comando e
carica ciascuno in D1, stessa logica per tutti (5 periodi standard,
idempotente, formato SQL identico).

Uso: python load_ohlc_generic.py SMI,IBEX35
     python load_ohlc_generic.py ITALY40

SICUREZZA: solo simboli presenti in SYMBOL_MAP sono accettati — nessuna
costruzione dinamica del nome costante Dukascopy a partire da input
utente (rischio: nome sbagliato carica dati del simbolo sbagliato senza
errore esplicito, come quasi successo con la prima ricerca SMI che ha
trovato solo azioni singole). Per aggiungere un nuovo simbolo:
  1. Eseguire discover_dukascopy_*.py per trovare/confermare la costante
  2. Aggiungere una riga a SYMBOL_MAP qui sotto
  3. Da quel momento il simbolo è disponibile per qualunque run futura,
     nessun nuovo script da scrivere.

Costanti già confermate nel progetto:
  ITALY40  -> INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR  (confermata, discover 14/07)
  SMI      -> INSTRUMENT_IDX_EUROPE_E_SWMI        (confermata, discover 15/07)
  IBEX35   -> INSTRUMENT_IDX_EUROPE_E_IBC_MAC     (trovata come sottoprodotto
              della ricerca SMI, NON ancora verificata con un discover
              dedicato — controllare il conteggio/range date al primo
              caricamento prima di fidarsi ciecamente)
  GBPUSD   -> INSTRUMENT_FX_MAJORS_GBP_USD        (aggiunta 22/07/2026 per
              il filone "identificazione regime FTSE100" — coppia forex
              standard, nome costante confermato via documentazione
              ufficiale dukascopy-python, NON ancora verificata con un
              caricamento reale — controllare conteggio/range date al
              primo run come per ogni simbolo nuovo)
"""

from __future__ import annotations

import sys
from datetime import datetime

import dukascopy_python
import dukascopy_python.instruments as instr

SYMBOL_MAP = {
    "ITALY40": "INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR",
    "SMI": "INSTRUMENT_IDX_EUROPE_E_SWMI",
    "IBEX35": "INSTRUMENT_IDX_EUROPE_E_IBC_MAC",  # non ancora verificata al 100%
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",  # non ancora verificata al 100%
}

CHUNK_SIZE = 500

PERIODS = {
    "2015-2016": (datetime(2015, 1, 1), datetime(2017, 1, 1)),
    "2020-covid": (datetime(2020, 1, 1), datetime(2021, 1, 1)),
    "2023": (datetime(2023, 1, 1), datetime(2024, 1, 1)),
    "2024-2025": (datetime(2024, 1, 1), datetime(2026, 1, 1)),
    "2026-ytd": (datetime(2026, 1, 1), datetime(2026, 7, 14)),
}


def fetch_period(instrument_const: str, start: datetime, end: datetime):
    instrument = getattr(instr, instrument_const)
    df = dukascopy_python.fetch(
        instrument,
        dukascopy_python.INTERVAL_MIN_30,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    return df.reset_index()


def load_symbol(symbol: str) -> tuple[str, int]:
    """Ritorna (nome_file_sql, n_righe). Genera insert_<symbol>.sql"""
    instrument_const = SYMBOL_MAP[symbol]
    all_rows = []
    total_expected = 0

    print(f"\n=== {symbol} (costante: {instrument_const}) ===")
    for label, (start, end) in PERIODS.items():
        print(f"  Scarico {label}: {start.date()} -> {end.date()}...")
        df = fetch_period(instrument_const, start, end)
        n = len(df)
        total_expected += n
        print(f"    {n} barre scaricate")

        if n == 0:
            print(f"    ATTENZIONE: zero barre per {label} — verificare manualmente.")
            continue

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

    print(f"  Totale barre raccolte per {symbol}: {len(all_rows)} (attese: {total_expected})")

    if not all_rows:
        print(f"  ERRORE: nessuna barra scaricata per {symbol}. Salto il file SQL.")
        return None, 0

    filename = f"insert_{symbol.lower()}.sql"
    with open(filename, "w") as f:
        f.write(f"DELETE FROM ohlc_prices WHERE symbol='{symbol}';\n\n")
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

    print(f"  File {filename} generato con {len(all_rows)} righe.")
    return filename, len(all_rows)


def main():
    if len(sys.argv) < 2:
        print("Uso: python load_ohlc_generic.py SIMBOLO1,SIMBOLO2,...")
        print(f"Simboli disponibili: {', '.join(SYMBOL_MAP.keys())}")
        sys.exit(1)

    requested = [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]
    unknown = [s for s in requested if s not in SYMBOL_MAP]
    if unknown:
        print(f"ERRORE: simboli non riconosciuti: {unknown}")
        print(f"Simboli disponibili (SYMBOL_MAP): {', '.join(SYMBOL_MAP.keys())}")
        print("Se è un simbolo nuovo, esegui prima un discover_dukascopy_*.py "
              "dedicato e aggiungi la costante confermata a SYMBOL_MAP.")
        sys.exit(1)

    generated_files = []
    for symbol in requested:
        filename, n_rows = load_symbol(symbol)
        if filename:
            generated_files.append(filename)

    # elenco per lo step successivo del workflow (uno per riga, letto da bash)
    with open("generated_sql_files.txt", "w") as f:
        f.write("\n".join(generated_files))

    print(f"\n=== Completato. {len(generated_files)} file SQL generati: {generated_files} ===")


if __name__ == "__main__":
    main()
