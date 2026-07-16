"""
live_signal_check.py — Check-segnale live per Fase 2 (demo).

Riusa `engine.generate_signals()` (INVARIATO, stesso codice del backtest)
su dati recenti DAX/FTSE100, individua se l'ultima barra da 30min CHIUSA
ha generato un segnale, e verifica se sarebbe AZIONABILE contro lo stato
reale in D1 (kill switch, ordini/giorno, posizioni concorrenti) —
esattamente le stesse regole del motore (`BacktestEngine.run()`, sez. 6).

QUESTO SCRIPT NON APRE ORDINI. Rileva e riporta solo. L'esecuzione reale
(modulo integrazione IG: auth, prezzi, invio ordini) è il passo
successivo della sequenza Fase 2, non ancora costruito — quando sarà
pronto, prenderà l'output di questo script (o la stessa logica) per
decidere se e cosa eseguire su IG.

Fonte prezzi: dukascopy_python (fetch per lo storico/warmup, live_fetch
per l'ultima barra — vedi nota approssimazione pre-IG nel riepilogo
progetto 15/07/2026). Questa approssimazione decade da sola quando il
modulo IG userà prezzi IG reali al posto di Dukascopy.

Pensato per girare via cron GitHub Actions ogni 30 minuti in orario di
mercato (workflow non ancora creato — punto 4 della sequenza).

Nessuna modifica a engine.py. Nessun ordine. Scrive solo in
`live_daily_state` (crea il record del giorno se mancante, non lo
modifica altrimenti) — non scrive in `live_positions`/`live_trades`,
quelli sono compiti del modulo di esecuzione (punto 3).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng

WARMUP_DAYS = 90
CAPITAL0_DEFAULT = 2000.0   # usato SOLO se non esiste ancora nessun live_daily_state
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

# Credenziali Cloudflare D1 (repository secrets, mai in chiaro)
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"


def d1_query(sql: str, params: list | None = None) -> list[dict]:
    """Esegue una query su D1 via REST API Cloudflare (nessun wrangler
    necessario, stessa tecnica già usata per gli altri workflow)."""
    import requests
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    body = {"sql": sql}
    if params:
        body["params"] = params
    resp = requests.post(url, json=body, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Query D1 fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_historical(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_latest_bar(symbol_const, day_start: datetime) -> tuple[pd.DataFrame, str]:
    """Usa live_fetch per l'ultima barra da 30min, più aggiornata di fetch()
    su intervalli non-tick (vedi nota progetto sulla fonte prezzi pre-IG).
    Prende il primo DataFrame utile dal generator e si ferma (esecuzione
    one-shot, non streaming continuo — questo script gira via cron).

    Ritorna (df, fonte) dove fonte è 'live_fetch' o 'fetch_fallback' —
    se live_fetch non restituisce nulla o solleva un'eccezione, ripiega
    su fetch() normale (più "delayed" sull'ultima barra ma comunque
    utilizzabile, meglio di saltare lo strumento del tutto).
    """
    time_unit = getattr(dukascopy_python, "TIME_UNIT_MIN", None) or \
                getattr(dukascopy_python, "TIME_UNIT_MINUTE", None)
    if time_unit is None:
        raise RuntimeError(
            "Costante TIME_UNIT_MIN/TIME_UNIT_MINUTE non trovata in dukascopy_python — "
            "verificare il nome esatto nella versione installata (pip show dukascopy-python) "
            "e correggere qui prima del primo run reale."
        )

    try:
        iterator = dukascopy_python.live_fetch(
            symbol_const, 30, time_unit, dukascopy_python.OFFER_SIDE_BID, day_start, None,
        )
        for df in iterator:
            df = df.reset_index()
            ts_col = df.columns[0]
            df = df.rename(columns={ts_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            if not df.empty:
                return df, "live_fetch"
            break  # generator ha dato un df vuoto, non insistere: passa al fallback
    except Exception as e:
        print(f"  [diagnostica] live_fetch ha sollevato un'eccezione: {type(e).__name__}: {e}")

    # fallback: fetch() normale sulle ultime ore, prendo la barra più recente disponibile
    print(f"  [diagnostica] live_fetch vuoto o fallito, ripiego su fetch() normale (dati più 'delayed').")
    fallback_start = datetime.now(timezone.utc) - timedelta(hours=6)
    fallback_end = datetime.now(timezone.utc) + timedelta(minutes=30)
    try:
        df = dukascopy_python.fetch(
            symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
            fallback_start, fallback_end,
        ).reset_index()
        ts_col = df.columns[0]
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, "fetch_fallback"
    except Exception as e:
        print(f"  [diagnostica] anche il fallback fetch() è fallito: {type(e).__name__}: {e}")
        return pd.DataFrame(), "nessuna_fonte"


CONSOLIDATE_PCT = 0.4      # opzione 3, decisa in chat 16/07/2026, validata su 5 periodi ufficiali
THRESHOLD_MULT = 1.5


def apply_monthly_consolidation_if_needed(today_str: str, prev_state: dict) -> dict:
    """Se il nuovo giorno inaugura un nuovo mese di calendario rispetto
    all'ultimo stato noto, applica il check di accantonamento (opzione 3
    mensile, validata su 5 periodi ufficiali il 16/07/2026) sul capitale
    di fine mese precedente. Ritorna i valori aggiornati da usare per il
    nuovo record del giorno. Se accantonamento_attivo=0, non fa nulla
    (interruttore per disattivarlo senza toccare il codice)."""
    if not prev_state.get("accantonamento_attivo", 1):
        return {
            "capital": prev_state["capital_current"],
            "accantonato": prev_state.get("accantonato", 0.0),
            "reference": prev_state.get("consolidamento_reference") or prev_state["capital_current"],
            "threshold": prev_state.get("consolidamento_threshold") or prev_state["capital_current"] * THRESHOLD_MULT,
        }

    capital = prev_state["capital_current"]
    accantonato = prev_state.get("accantonato", 0.0) or 0.0
    reference = prev_state.get("consolidamento_reference") or capital
    threshold = prev_state.get("consolidamento_threshold") or (reference * THRESHOLD_MULT)

    prev_month = prev_state["trade_date"][:7]  # 'YYYY-MM'
    this_month = today_str[:7]

    if this_month != prev_month:
        while capital > threshold:
            gain = capital - reference
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            accantonato += consolidated
            capital -= consolidated
            reference = capital
            threshold = reference * THRESHOLD_MULT
            print(f"[accantonamento] Consolidati {consolidated:.2f} EUR al cambio mese "
                  f"({prev_month} -> {this_month}). Investito: {capital:.2f}  Accantonato: {accantonato:.2f}")

    return {"capital": capital, "accantonato": accantonato, "reference": reference, "threshold": threshold}


def get_or_create_today_state(today_str: str) -> dict:
    rows = d1_query(f"SELECT * FROM live_daily_state WHERE trade_date = '{today_str}'")
    if rows:
        return rows[0]

    prev_rows = d1_query(
        "SELECT * FROM live_daily_state ORDER BY trade_date DESC LIMIT 1"
    )

    if prev_rows:
        prev_state = prev_rows[0]
        updated = apply_monthly_consolidation_if_needed(today_str, prev_state)
        starting_capital = updated["capital"]
        starting_accantonato = updated["accantonato"]
        reference = updated["reference"]
        threshold = updated["threshold"]
    else:
        starting_capital = CAPITAL0_DEFAULT
        starting_accantonato = 0.0
        reference = CAPITAL0_DEFAULT
        threshold = CAPITAL0_DEFAULT * THRESHOLD_MULT

    d1_query(
        "INSERT INTO live_daily_state "
        "(trade_date, account_type, capital_start_of_day, capital_current, accantonato, "
        "consolidamento_reference, consolidamento_threshold, accantonamento_attivo) "
        f"VALUES ('{today_str}', 'demo', {starting_capital}, {starting_capital}, {starting_accantonato}, "
        f"{reference}, {threshold}, 1)"
    )
    print(f"Creato nuovo record live_daily_state per {today_str}: "
          f"investito={starting_capital:.2f} EUR, accantonato={starting_accantonato:.2f} EUR")
    return {
        "trade_date": today_str, "account_type": "demo",
        "capital_start_of_day": starting_capital, "capital_current": starting_capital,
        "accantonato": starting_accantonato, "orders_today": 0, "kill_switch_triggered": 0,
    }


def get_open_positions() -> list[dict]:
    return d1_query("SELECT * FROM live_positions WHERE status = 'open'")


def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN mancanti.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    warmup_start = day_start - timedelta(days=WARMUP_DAYS)

    print(f"=== Check segnale live — {now.isoformat()} ===\n")

    # 1) stato giornata (crea se mancante, MAI sovrascrive se già esiste)
    day_state = get_or_create_today_state(today_str)
    kill_switch_active = bool(day_state["kill_switch_triggered"])
    orders_today = day_state["orders_today"]
    print(f"Stato giornata: investito={day_state['capital_current']:.2f} EUR, "
          f"accantonato={day_state.get('accantonato', 0.0):.2f} EUR, "
          f"ordini oggi={orders_today}/{eng.PARAMS.max_new_orders_per_day}, "
          f"kill switch attivo={kill_switch_active}\n")

    # 2) posizioni aperte (per vincolo max concorrenti + niente 2 sullo stesso strumento)
    open_positions = get_open_positions()
    open_instruments = {p["instrument"] for p in open_positions}
    print(f"Posizioni aperte: {len(open_positions)}/{eng.PARAMS.max_concurrent_positions} "
          f"({', '.join(open_instruments) if open_instruments else 'nessuna'})\n")

    # 3) segnali per strumento
    detected = []
    for name, const in SYMBOLS.items():
        inst = eng.INSTRUMENTS[name]
        hist = fetch_historical(const, warmup_start, day_start)
        latest, source = fetch_latest_bar(const, day_start)

        if latest.empty:
            print(f"{name}: nessuna barra disponibile da nessuna fonte (live_fetch e fallback entrambi vuoti), salto.")
            continue
        if source == "fetch_fallback":
            print(f"{name}: uso dati di fallback (fetch normale, possibile ritardo sull'ultima barra)")

        combined = pd.concat([hist, latest]).drop_duplicates(subset="timestamp") \
                       .sort_values("timestamp").reset_index(drop=True)
        signals = eng.generate_signals(combined, inst)

        # ultima barra CHIUSA: quella con timestamp + 30min <= now
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            print(f"{name}: nessuna barra chiusa disponibile ancora, salto.")
            continue

        last_closed = closed.iloc[-1]
        sig = last_closed["signal"]
        print(f"{name}: ultima barra chiusa {last_closed['timestamp']} — segnale: {sig or 'nessuno'}")

        if sig in ("long", "short"):
            already_open = name in open_instruments
            detected.append({
                "instrument": name, "direction": sig,
                "bar_time": last_closed["timestamp"],
                "atr": last_closed["atr"], "adx": last_closed["adx"],
                "already_open": already_open,
            })

    if not detected:
        print("\nNessun segnale nuovo su questa barra. Fine check.")
        return

    # 4) verifica azionabilità (stesse regole di BacktestEngine.run())
    print(f"\n=== {len(detected)} segnale/i rilevato/i — verifica azionabilità ===")
    slots_free = eng.PARAMS.max_concurrent_positions - len(open_positions)

    for d in detected:
        reasons_blocking = []
        if kill_switch_active:
            reasons_blocking.append("kill switch attivo oggi")
        if orders_today >= eng.PARAMS.max_new_orders_per_day:
            reasons_blocking.append(f"limite ordini/giorno raggiunto ({orders_today}/{eng.PARAMS.max_new_orders_per_day})")
        if d["already_open"]:
            reasons_blocking.append(f"posizione già aperta su {d['instrument']}")
        if slots_free <= 0:
            reasons_blocking.append(f"nessuno slot concorrente libero ({len(open_positions)}/{eng.PARAMS.max_concurrent_positions})")
        if pd.isna(d["atr"]) or pd.isna(d["adx"]):
            reasons_blocking.append("ATR/ADX non disponibili sulla barra segnale")

        azionabile = len(reasons_blocking) == 0
        stato = "AZIONABILE" if azionabile else "BLOCCATO"
        print(f"\n{d['instrument']} {d['direction'].upper()} (barra {d['bar_time']}): {stato}")
        if reasons_blocking:
            for r in reasons_blocking:
                print(f"  - {r}")
        else:
            print(f"  ATR={d['atr']:.2f}  ADX={d['adx']:.1f}")
            print("  -> pronto per l'esecuzione (modulo IG non ancora collegato: "
                  "nessun ordine inviato da questo script).")
            if slots_free > 0:
                slots_free -= 1  # simula occupazione slot per eventuali segnali multipli nello stesso run

    print("\n=== Check completato. Nessun ordine inviato (script di sola rilevazione). ===")


if __name__ == "__main__":
    main()
