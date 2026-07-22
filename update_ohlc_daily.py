"""
update_ohlc_daily.py — Aggiornamento incrementale giornaliero di
ohlc_prices per DAX, FTSE100, GOLD, GBPUSD. Decisione 22/07/2026:
scoperto che NESSUN job esistente tiene aggiornato ohlc_prices — nemmeno
live_execute.py (gira ogni 30min ma scarica DAX/FTSE100 direttamente da
Dukascopy IN MEMORIA per calcolare i segnali, senza mai scrivere in
ohlc_prices). Finora ohlc_prices veniva aggiornato solo come effetto
collaterale di script di ricerca che chiamavano get_ohlc() — copertura
irregolare, non garantita.

Questo script chiude quel buco: chiama get_ohlc() (stessa funzione
condivisa di ohlc_data_source.py, aggiornamento incrementale — scarica
solo le barre mancanti da Dukascopy, non l'intero storico) per tutti e
4 i simboli, una volta al giorno. Volume atteso per esecuzione: poche
decine/centinaia di righe per simbolo (solo le barre del giorno
precedente) — ben sotto qualunque limite di scrittura D1.

Nessuna scrittura oltre a quella già gestita da get_ohlc() (mai UPDATE/
DELETE, solo INSERT OR IGNORE di righe nuove). Nessuna modifica alla
logica di trading (live_execute.py resta invariato, continua a
scaricare in memoria per i segnali — questo script è indipendente e
serve solo a mantenere il dataset di ricerca aggiornato).

Trigger: cron-job.org esterno che chiama workflow_dispatch via API
GitHub, MAI lo schedule nativo di GitHub Actions — stesso pattern già
stabilito per live_execute_cron e per il cron dei candidati (schedule
nativo GitHub Actions si è dimostrato inaffidabile, si fermava dopo un
trigger manuale).
"""

from __future__ import annotations

import os

from ohlc_data_source import get_ohlc, DUKASCOPY_CONST


def main():
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]

    print(f"=== Aggiornamento incrementale ohlc_prices — simboli: {list(DUKASCOPY_CONST)} ===\n")

    for symbol in DUKASCOPY_CONST:
        print(f"--- {symbol} ---")
        try:
            df = get_ohlc(symbol, account_id, token)
            print(f"  [{symbol}] OK — serie aggiornata, {len(df)} righe totali, "
                  f"ultimo dato: {df['timestamp'].max().isoformat()}\n")
        except Exception as e:
            print(f"  [{symbol}] ERRORE durante l'aggiornamento: {e}\n")

    print("=== Completato. ===")


if __name__ == "__main__":
    main()
