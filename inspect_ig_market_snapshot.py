"""
inspect_ig_market_snapshot.py — Ispeziona la risposta COMPLETA
dell'endpoint /markets/{epic} di IG, invece di limitarsi ai campi
bid/offer già estratti in ig_client.py. Serve a rispondere con
certezza (non a intuito) alla domanda: IG espone anche volume e/o
indicatori di liquidità oltre al prezzo?

Nessun ordine. Solo lettura, stampa il JSON completo (troncato se
molto lungo) per ispezione manuale.
"""

import json
from ig_client import IGSession, load_credentials_from_env, EPIC_MAP
import requests

BASE_URL = "https://demo-api.ig.com/gateway/deal"


def main():
    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        for instrument, epic in EPIC_MAP.items():
            print(f"\n=== {instrument} ({epic}) — risposta completa /markets/{{epic}} ===\n")
            resp = requests.get(f"{BASE_URL}/markets/{epic}", headers=session._headers(version="3"), timeout=15)
            resp.raise_for_status()
            data = resp.json()

            print("--- Campi in 'snapshot' (prezzo/mercato) ---")
            print(json.dumps(data.get("snapshot", {}), indent=2))

            print("\n--- Campi in 'instrument' (dettagli strumento) ---")
            instrument_data = data.get("instrument", {})
            # stampo solo le chiavi per non intasare il log, poi i valori di quelle sospette
            print(f"Chiavi disponibili: {list(instrument_data.keys())}")
            for key in instrument_data:
                if any(term in key.lower() for term in ["volume", "liquid", "depth", "size"]):
                    print(f"  {key}: {instrument_data[key]}")

            print("\n--- Campi in 'dealingRules' ---")
            dealing_rules = data.get("dealingRules", {})
            print(json.dumps(dealing_rules, indent=2))

    print("\n=== Fine ispezione. Nessun ordine inviato. ===")


if __name__ == "__main__":
    main()
