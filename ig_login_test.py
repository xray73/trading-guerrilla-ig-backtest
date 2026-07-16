"""
ig_login_test.py — Verifica SOLO l'accesso via API al conto demo IG.
Fa un login (POST /session), controlla che risponda con i token di
sessione (CST, X-SECURITY-TOKEN), poi chiude subito la sessione
(DELETE /session). Nessun ordine, nessuna lettura di posizioni/prezzi
oltre al login stesso — puro test di connettività/credenziali.

Credenziali lette da variabili d'ambiente (impostate dal workflow a
partire dai repository secrets, mai in chiaro né in questo file né in
chat):
  IG_API_KEY       -> secret TRADING_GUERRILLA_DEMO
  IG_DEMO_USERNAME -> secret IG_DEMO_USERNAME
  IG_DEMO_PASSWORD -> secret ID_DEMO_PASSWORD

Endpoint: demo-api.ig.com (SEMPRE demo in questa fase del progetto,
mai api.ig.com live).
"""

import os
import sys
import requests

BASE_URL = "https://demo-api.ig.com/gateway/deal"


def mask(s: str, keep: int = 4) -> str:
    if not s:
        return "(vuoto)"
    return s[:keep] + "…" + f"({len(s)} char totali)"


def main():
    api_key = os.environ.get("IG_API_KEY")
    username = os.environ.get("IG_DEMO_USERNAME")
    password = os.environ.get("IG_DEMO_PASSWORD")

    missing = [name for name, val in [("IG_API_KEY", api_key),
                                        ("IG_DEMO_USERNAME", username),
                                        ("IG_DEMO_PASSWORD", password)] if not val]
    if missing:
        print(f"ERRORE: variabili d'ambiente mancanti: {missing}")
        print("Controlla che i repository secrets siano mappati correttamente nel workflow.")
        sys.exit(1)

    print(f"API key:  {mask(api_key)}")
    print(f"Username: {mask(username, keep=2)}")
    print(f"Password: {mask(password, keep=0)}")
    print(f"\nTento login su {BASE_URL}/session (conto DEMO)...\n")

    headers = {
        "X-IG-API-KEY": api_key,
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "VERSION": "2",
    }
    body = {"identifier": username, "password": password}

    try:
        resp = requests.post(f"{BASE_URL}/session", json=body, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"ERRORE DI RETE: {e}")
        sys.exit(1)

    print(f"Status HTTP: {resp.status_code}")

    if resp.status_code != 200:
        print("LOGIN FALLITO.")
        try:
            print(f"Dettaglio errore IG: {resp.json()}")
        except Exception:
            print(f"Risposta grezza: {resp.text[:500]}")
        sys.exit(1)

    cst = resp.headers.get("CST")
    security_token = resp.headers.get("X-SECURITY-TOKEN")
    data = resp.json()

    if not cst or not security_token:
        print("LOGIN APPARENTEMENTE OK (200) ma token di sessione mancanti nell'header — anomalo.")
        print(f"Body risposta: {data}")
        sys.exit(1)

    print("LOGIN RIUSCITO.")
    print(f"  Account ID: {data.get('currentAccountId', 'n/d')}")
    print(f"  Client ID:  {data.get('clientId', 'n/d')}")
    print(f"  Lightstreamer endpoint: {data.get('lightstreamerEndpoint', 'n/d')}")
    print(f"  CST: {mask(cst)}")
    print(f"  X-SECURITY-TOKEN: {mask(security_token)}")

    # chiusura pulita della sessione (buona pratica, libera lo slot di sessione IG)
    close_headers = {
        "X-IG-API-KEY": api_key,
        "CST": cst,
        "X-SECURITY-TOKEN": security_token,
        "VERSION": "1",
    }
    try:
        close_resp = requests.delete(f"{BASE_URL}/session", headers=close_headers, timeout=15)
        print(f"\nChiusura sessione: status {close_resp.status_code} "
              f"({'OK' if close_resp.status_code in (200, 204) else 'verificare manualmente'})")
    except requests.exceptions.RequestException as e:
        print(f"\nAttenzione: chiusura sessione fallita ({e}) — non critico, la sessione scade comunque da sola.")

    print("\n=== TEST COMPLETATO: accesso API al conto demo confermato. ===")


if __name__ == "__main__":
    main()
