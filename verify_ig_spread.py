"""
verify_ig_spread.py — Verifica lo spread reale IG durante l'orario di
mercato contro lo spread fisso assunto dal motore (engine.py,
spread_fixed, esplicitamente marcato "da riverificare" nel Charter).

Valori assunti dal motore: DAX 1.2 punti, FTSE100 1.0 punti.

Un singolo campione non basta — lo spread varia con la liquidità
(più stretto a metà giornata di alto volume, più largo in apertura/
chiusura o momenti di bassa liquidità). Questo script salva un
campione ad ogni esecuzione in D1 (tabella spread_samples); va
lanciato più volte nell'arco della giornata (es. agganciato allo
stesso cron di live_execute.py) per costruire una distribuzione
prima di trarre conclusioni da un singolo numero.

Nessun ordine. Solo lettura prezzi IG + scrittura diagnostica in D1.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from ig_client import IGSession, load_credentials_from_env

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

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
    print(f"=== Campionamento spread IG — {now.isoformat()} ===\n")

    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        for instrument in ("DAX", "FTSE100"):
            try:
                price = session.get_price(instrument)
            except Exception as e:
                print(f"  [{instrument}] impossibile leggere il prezzo ({e})")
                continue

            bid, offer = price["bid"], price["offer"]
            if bid is None or offer is None:
                print(f"  [{instrument}] prezzo non disponibile (bid/offer None) "
                      f"— mercato probabilmente chiuso o strumento non tradeable ora.")
                continue

            spread = offer - bid
            assumed = ASSUMED_SPREAD[instrument]
            delta = spread - assumed
            print(f"  [{instrument}] bid={bid} offer={offer} spread_reale={spread:.2f}pt "
                  f"assunto={assumed}pt scarto={delta:+.2f}pt stato={price['market_status']}")

            d1_query(
                "INSERT INTO spread_samples (instrument, sample_time, bid, offer, spread, market_status) "
                f"VALUES ('{instrument}', '{now.isoformat()}', {bid}, {offer}, {spread}, "
                f"'{price['market_status']}')"
            )

    print("\n=== Completato. Campione salvato in D1 (tabella spread_samples). ===")
    print("Serve accumulare più campioni nel tempo prima di trarre conclusioni — "
          "vedi verify_ig_spread_analysis.py per l'analisi aggregata.")


if __name__ == "__main__":
    main()
