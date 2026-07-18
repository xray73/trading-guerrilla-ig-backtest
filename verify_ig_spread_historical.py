"""
verify_ig_spread_historical.py — Stima lo spread reale IG usando lo
storico bid/ask (endpoint /prices/{epic} di IG, già esposto da
ig_client.get_historical_prices(), NESSUNA modifica a ig_client.py).

DIFFERENZA rispetto a verify_ig_spread.py: quello cattura UN singolo
snapshot live per esecuzione (n cresce di 1 ogni volta che gira,
lentamente nel tempo). Questo script scarica DECINE di barre storiche
bid/ask in una sola chiamata — utile per stimare rapidamente lo
spread ATTUALE (ultimi giorni/settimane) con un campione molto più
solido, ma NON dà lo spread storico dei 5 periodi di backtest
ufficiali (2015-2026) — quello non è recuperabile, lo spread reale di
allora non è mai stato registrato da nessuna fonte disponibile, e
comunque cambia nel tempo con la liquidità. Serve solo a calibrare
meglio spread_fixed da qui in avanti, non a rifare i backtest storici
con spread accurato per periodo.

LIMITE QUOTA: l'endpoint storico IG consuma la quota settimanale
(~10.000 punti dato, reset ogni 7 giorni — condivisa con
get_historical_prices() usato altrove, es. eventuali test manuali).
Finestra di default qui: ultimi 5 giorni di mercato, risoluzione
30min (~48 barre/giorno) = ~240 punti per strumento, ~480 totali —
trascurabile rispetto alla quota, ripetibile senza problemi.

Metodologia: spread = ask - bid sul prezzo di CHIUSURA di ogni barra
storica (rappresentativo, evita di pesare 4x per barra con
open/high/low/close). Aggregato per strumento e per fascia oraria UTC
(per verificare l'ipotesi già sospettata: spread più stretto a metà
giornata, più largo agli estremi di sessione).

Salva ogni punto in D1 (tabella spread_samples esistente, nessuna
modifica di schema) con market_status='HISTORICAL' per distinguerli
dagli snapshot live di verify_ig_spread.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from ig_client import IGSession, load_credentials_from_env

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

INSTRUMENTS = ("DAX", "FTSE100")
LOOKBACK_DAYS = 5          # finestra storica, conservativa sulla quota
RESOLUTION = "MINUTE_30"   # stessa granularità del motore (30min)
ASSUMED_SPREAD = {"DAX": 1.2, "FTSE100": 1.0}  # da engine.py, spread_fixed


def d1_query(sql: str) -> list[dict]:
    import requests
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Query D1 fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN mancanti.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)
    print(f"=== Stima spread storica IG — finestra {start.isoformat()} -> {now.isoformat()} ===\n")

    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        for instrument in INSTRUMENTS:
            print(f"--- {instrument} ---")
            try:
                data = session.get_historical_prices(
                    instrument, RESOLUTION,
                    start.strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S"),
                )
            except Exception as e:
                print(f"  ERRORE nel download storico: {e}")
                continue

            allowance = data.get("allowance", {})
            print(f"  Quota residua dopo questa chiamata: {allowance.get('remainingAllowance', '?')} "
                  f"/ {allowance.get('totalAllowance', '?')} (reset tra {allowance.get('allowanceExpiry', '?')}s)")

            prices = data.get("prices", [])
            if not prices:
                print("  Nessuna barra storica ritornata — verificare finestra/quota.")
                continue

            spreads = []
            by_hour = defaultdict(list)
            insert_values = []

            for bar in prices:
                close = bar.get("closePrice", {})
                bid, ask = close.get("bid"), close.get("ask")
                if bid is None or ask is None:
                    continue
                spread = ask - bid
                ts = bar.get("snapshotTimeUTC") or bar.get("snapshotTime")
                spreads.append(spread)
                try:
                    hour = int(str(ts)[11:13])
                    by_hour[hour].append(spread)
                except (ValueError, TypeError):
                    pass
                if ts:
                    insert_values.append((ts, bid, ask, spread))

            if not spreads:
                print("  Barre ricevute ma nessuna con bid/ask valorizzati (mercato chiuso in tutta la finestra?).")
                continue

            spreads_sorted = sorted(spreads)
            n = len(spreads_sorted)
            mean_spread = sum(spreads) / n
            median_spread = spreads_sorted[n // 2]
            assumed = ASSUMED_SPREAD[instrument]

            print(f"  Barre valide: {n}")
            print(f"  Spread — media: {mean_spread:.2f}pt  mediana: {median_spread:.2f}pt  "
                  f"min: {min(spreads):.2f}pt  max: {max(spreads):.2f}pt")
            print(f"  Assunto nel motore (spread_fixed): {assumed}pt  "
                  f"scarto medio: {mean_spread - assumed:+.2f}pt")

            print("  Per fascia oraria UTC (media):")
            for hour in sorted(by_hour):
                vals = by_hour[hour]
                print(f"    {hour:02d}:00 — media {sum(vals)/len(vals):.2f}pt (n={len(vals)})")

            for ts, bid, ask, spread in insert_values:
                d1_query(
                    "INSERT INTO spread_samples (instrument, sample_time, bid, offer, spread, market_status) "
                    f"VALUES ('{instrument}', '{ts}', {bid}, {ask}, {spread}, 'HISTORICAL')"
                )
            print(f"  Salvati {len(insert_values)} punti storici in D1 (spread_samples, market_status='HISTORICAL').\n")

    print("=== Completato. ===")


if __name__ == "__main__":
    main()
