"""
check_volume_quality.py — Verifica se il campo volume di Dukascopy su
DAX/FTSE100 è utilizzabile per calcolare un VWAP (prerequisito per la
specifica ORB+VWAP, ramo esplorativo). Analisi diagnostica pura,
nessuna modifica a nulla.

Controlla, su un campione recente (30 giorni) a risoluzione fine
(1min, quella rilevante per un VWAP di sessione):
  - % di barre con volume = 0 o NaN
  - distribuzione (il volume varia in modo sensato durante la giornata,
    es. più alto in apertura/chiusura, o è piatto/rumoroso senza pattern?)
  - confronto: il volume "spiega" qualcosa, o è solo un contatore di tick
    che Dukascopy sintetizza e non riflette liquidità reale?
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
DAYS_BACK = 30


def fetch_1min(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_1, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)

    print(f"=== Verifica qualità volume Dukascopy — ultimi {DAYS_BACK} giorni, barre 1min ===\n")
    print(f"Colonne attese: verificare se esiste una colonna 'volume' nel dataframe.\n")

    for name, const in SYMBOLS.items():
        print(f"--- {name} ---")
        df = fetch_1min(const, start, end)
        print(f"Colonne disponibili: {list(df.columns)}")

        if "volume" not in df.columns:
            print("NESSUNA colonna volume disponibile — VWAP non calcolabile con questi dati. Fine controllo per questo strumento.\n")
            continue

        n = len(df)
        n_zero = (df["volume"] == 0).sum()
        n_nan = df["volume"].isna().sum()
        print(f"Barre totali: {n}")
        print(f"Barre con volume=0: {n_zero} ({100*n_zero/n:.1f}%)")
        print(f"Barre con volume=NaN: {n_nan} ({100*n_nan/n:.1f}%)")
        print(f"Volume medio: {df['volume'].mean():.2f}  mediana: {df['volume'].median():.2f}  "
              f"std: {df['volume'].std():.2f}")

        # pattern intraday: il volume medio per ora del giorno mostra la forma a "U"
        # tipica (alto in apertura/chiusura, basso a metà giornata)? Se sì, il dato
        # ha senso economico; se piatto/casuale, è probabile sia solo un tick-count
        # sintetico poco informativo.
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour")["volume"].mean().sort_index()
        print("Volume medio per ora UTC (cerca pattern a U, alto ai bordi):")
        for h, v in hourly.items():
            bar = "#" * int(v / hourly.max() * 40) if hourly.max() > 0 else ""
            print(f"  {h:02d}:00  {v:8.2f}  {bar}")
        print()

    print("=== Fine diagnostica. Nessuna modifica effettuata. ===")


if __name__ == "__main__":
    main()
