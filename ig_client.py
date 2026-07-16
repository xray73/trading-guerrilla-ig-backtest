"""
ig_client.py — Modulo di integrazione IG per Fase 2 (demo trading).

Fornisce: autenticazione (riusa la logica già testata in ig_login_test.py),
lettura prezzi correnti, invio ordine (posizione + stop/target), chiusura
posizione. SEMPRE contro demo-api.ig.com in questa fase — mai api.ig.com
(live) finché non deciso esplicitamente e verificato più volte.

Design: funzioni pure/stateless dove possibile, una classe IGSession che
gestisce il ciclo di vita di CST/X-SECURITY-TOKEN per non fare login ad
ogni chiamata. Pensato per essere importato da live_signal_check.py (o dal
suo successore che unisce check+esecuzione) e dal workflow cron.

Nessun ordine viene mai piazzato "a scatola chiusa": ogni funzione di
scrittura (place_order, close_position) logga esplicitamente cosa sta per
fare PRIMA di farlo, e tutte le chiamate finiscono comunque in
live_positions/live_trades su D1 per audit completo.

Mapping strumento -> EPIC IG: da verificare e completare con gli EPIC
reali (Impostazioni -> Cerca mercato -> DAX/FTSE100 -> "Dettagli" mostra
l'EPIC). Placeholder sotto — NON usare in produzione senza aver confermato
gli EPIC corretti dalla piattaforma IG.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Literal
import requests

BASE_URL = "https://demo-api.ig.com/gateway/deal"

# TODO: verificare gli EPIC esatti da IG (Cerca mercato -> Dettagli) prima
# di usare questo modulo per ordini reali. Questi sono placeholder plausibili
# basati sulla nomenclatura standard IG, NON confermati.
EPIC_MAP = {
    "DAX": "IX.D.DAX.IFMM.IP",       # verificato funzionante (prezzo letto correttamente)
    "FTSE100": "IX.D.FTSE.IFE.IP",   # corretto il 16/07/2026: "FTSE 100 Cash (1€)", trovato via search_markets
}


@dataclass
class IGCredentials:
    api_key: str
    username: str
    password: str


class IGSession:
    """Gestisce login/logout e tiene i token di sessione per riuso tra
    più chiamate nello stesso run (evita un login per ogni richiesta)."""

    def __init__(self, creds: IGCredentials):
        self.creds = creds
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self.account_id: Optional[str] = None

    def _headers(self, version: str = "2") -> dict:
        h = {
            "X-IG-API-KEY": self.creds.api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "VERSION": version,
        }
        if self.cst and self.security_token:
            h["CST"] = self.cst
            h["X-SECURITY-TOKEN"] = self.security_token
        return h

    def login(self) -> None:
        body = {"identifier": self.creds.username, "password": self.creds.password}
        resp = requests.post(f"{BASE_URL}/session", json=body, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        self.cst = resp.headers.get("CST")
        self.security_token = resp.headers.get("X-SECURITY-TOKEN")
        data = resp.json()
        self.account_id = data.get("currentAccountId")
        if not self.cst or not self.security_token:
            raise RuntimeError("Login riuscito (200) ma token di sessione mancanti — anomalo.")

    def logout(self) -> None:
        if not self.cst:
            return
        try:
            requests.delete(f"{BASE_URL}/session", headers=self._headers(version="1"), timeout=15)
        except requests.exceptions.RequestException:
            pass  # non critico, la sessione scade comunque da sola
        self.cst = None
        self.security_token = None

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()

    def search_markets(self, search_term: str) -> list[dict]:
        """Cerca mercati per nome (es. 'FTSE 100', 'Germany 40') e
        ritorna epic/nome/tipo di ciascun risultato — modo affidabile
        per trovare l'EPIC esatto invece di cercarlo a mano nell'interfaccia."""
        resp = requests.get(f"{BASE_URL}/markets", params={"searchTerm": search_term},
                             headers=self._headers(version="1"), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [
            {"epic": m.get("epic"), "instrumentName": m.get("instrumentName"),
             "instrumentType": m.get("instrumentType"), "expiry": m.get("expiry")}
            for m in data.get("markets", [])
        ]

    # ---- lettura prezzi ----

    def get_price(self, instrument: str) -> dict:
        """Ritorna il prezzo corrente (bid/offer) per lo strumento.
        Solleva ValueError se l'EPIC non è mappato/verificato."""
        epic = EPIC_MAP.get(instrument)
        if not epic:
            raise ValueError(f"Nessun EPIC mappato per '{instrument}' — verificare EPIC_MAP.")

        resp = requests.get(f"{BASE_URL}/markets/{epic}", headers=self._headers(version="3"), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        snapshot = data.get("snapshot", {})
        return {
            "instrument": instrument,
            "epic": epic,
            "bid": snapshot.get("bid"),
            "offer": snapshot.get("offer"),
            "market_status": snapshot.get("marketStatus"),
            "update_time": snapshot.get("updateTime"),
        }

    # ---- ordini ----

    def place_order(self, instrument: str, direction: Literal["BUY", "SELL"],
                     size: float, stop_distance: float, limit_distance: float,
                     dry_run: bool = True) -> dict:
        """Apre una posizione con stop/target relativi (in punti dal prezzo
        di apertura). dry_run=True di default: NON invia l'ordine, ritorna
        solo cosa verrebbe inviato — va passato esplicitamente False per
        eseguire davvero. Questa è una misura di sicurezza intenzionale,
        non rimuoverla senza discuterne."""
        epic = EPIC_MAP.get(instrument)
        if not epic:
            raise ValueError(f"Nessun EPIC mappato per '{instrument}' — verificare EPIC_MAP.")

        payload = {
            "epic": epic,
            "expiry": "-",
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "guaranteedStop": False,
            "stopDistance": str(stop_distance),
            "limitDistance": str(limit_distance),
            "forceOpen": True,
            "currencyCode": "EUR",
        }

        print(f"[ig_client] Ordine {'SIMULATO (dry_run)' if dry_run else 'REALE'}: "
              f"{direction} {size} {instrument} ({epic}), stop={stop_distance}pt, target={limit_distance}pt")

        if dry_run:
            return {"status": "dry_run", "payload": payload}

        resp = requests.post(f"{BASE_URL}/positions/otc", json=payload,
                              headers=self._headers(version="2"), timeout=15)
        resp.raise_for_status()
        deal_ref = resp.json().get("dealReference")

        # conferma (IG richiede una chiamata separata per il risultato definitivo)
        confirm = requests.get(f"{BASE_URL}/confirms/{deal_ref}", headers=self._headers(version="1"), timeout=15)
        confirm.raise_for_status()
        return confirm.json()

    def close_position(self, deal_id: str, direction: Literal["BUY", "SELL"],
                        size: float, dry_run: bool = True) -> dict:
        """Chiude una posizione esistente. direction qui è la direzione di
        CHIUSURA (opposta a quella di apertura). dry_run=True di default,
        stessa logica di sicurezza di place_order."""
        payload = {
            "dealId": deal_id,
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
        }
        print(f"[ig_client] Chiusura {'SIMULATA (dry_run)' if dry_run else 'REALE'}: "
              f"dealId={deal_id}, {direction} {size}")

        if dry_run:
            return {"status": "dry_run", "payload": payload}

        resp = requests.delete(f"{BASE_URL}/positions/otc", json=payload,
                                headers=self._headers(version="1"), timeout=15)
        resp.raise_for_status()
        return resp.json()


def load_credentials_from_env() -> IGCredentials:
    api_key = os.environ.get("IG_API_KEY")
    username = os.environ.get("IG_DEMO_USERNAME")
    password = os.environ.get("IG_DEMO_PASSWORD")
    missing = [n for n, v in [("IG_API_KEY", api_key), ("IG_DEMO_USERNAME", username),
                                ("IG_DEMO_PASSWORD", password)] if not v]
    if missing:
        print(f"ERRORE: variabili d'ambiente mancanti: {missing}")
        sys.exit(1)
    return IGCredentials(api_key=api_key, username=username, password=password)


if __name__ == "__main__":
    # self-test: login, lettura prezzo DAX e FTSE100 (EPIC entrambi
    # verificati il 16/07/2026), nessun ordine.
    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        print(f"Sessione OK, account {session.account_id}")
        for instrument in ("DAX", "FTSE100"):
            try:
                price = session.get_price(instrument)
                print(f"Prezzo {instrument}: bid={price['bid']} offer={price['offer']} "
                      f"stato_mercato={price['market_status']}")
            except Exception as e:
                print(f"ATTENZIONE: lettura prezzo {instrument} fallita ({type(e).__name__}: {e})")
