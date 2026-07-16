"""
live_execute.py — Collega check-segnale, sizing (rispettando
l'accantonamento in D1) e ig_client per l'esecuzione su IG demo.
Sostituisce live_signal_check.py come script eseguito dal cron una
volta testato (per ora resta un file separato, non sovrascrive quello
esistente).

DRY_RUN=True di default (variabile d'ambiente DRY_RUN, default "true")
— nessun ordine reale finché non viene impostata esplicitamente a
"false". Misura di sicurezza intenzionale, stessa logica di
ig_client.place_order/close_position.

Ciclo ad ogni esecuzione (pensato per girare via cron ogni 30min):
  1. Gestisce le posizioni aperte: legge il prezzo corrente da IG,
     controlla se stop/target sono stati toccati o se il max holding
     è scaduto, chiude su IG se serve, aggiorna live_positions/
     live_trades/live_daily_state.capital_current.
  2. Rileva nuovi segnali (stessa logica di live_signal_check.py),
     calcola il size usando capital_current (che riflette
     l'accantonamento — l'accantonato è già fuori da questo numero),
     invia l'ordine a IG (o lo simula in dry_run), registra la
     posizione aperta.

Sizing: stessa formula del motore (_position_size in engine.py) —
risk_amount = capital_current * risk_pct, size = risk_amount /
(stop_distance_in_punti * point_value), forzato al minimo se sotto
soglia. Nessuna divergenza dalla logica già validata nel backtest.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from ig_client import IGSession, load_credentials_from_env

WARMUP_DAYS = 90
CAPITAL0_DEFAULT = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"


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


def fetch_historical(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_today_state(today_str: str) -> dict:
    rows = d1_query(f"SELECT * FROM live_daily_state WHERE trade_date = '{today_str}'")
    if not rows:
        raise RuntimeError(
            f"Nessuno stato per {today_str} in live_daily_state — esegui prima "
            f"live_signal_check.py (che lo crea) o inseriscilo manualmente."
        )
    return rows[0]


def manage_open_positions(session: IGSession, today_str: str):
    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    if not open_positions:
        print("Nessuna posizione aperta da gestire.")
        return

    day_state = get_today_state(today_str)
    capital = day_state["capital_current"]

    for pos in open_positions:
        try:
            price = session.get_price(pos["instrument"])
        except Exception as e:
            print(f"  [{pos['instrument']}] impossibile leggere il prezzo ({e}), salto.")
            continue

        current_price = price["bid"] if pos["direction"] == "long" else price["offer"]
        if current_price is None:
            print(f"  [{pos['instrument']}] prezzo non disponibile, salto.")
            continue

        exit_reason = None
        if pos["direction"] == "long":
            if current_price <= pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price >= pos["take_profit"]:
                exit_reason = "take_profit"
        else:
            if current_price >= pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= pos["take_profit"]:
                exit_reason = "take_profit"

        entry_time = pd.Timestamp(pos["entry_time"])
        max_holding_bars = pos.get("max_holding_bars") or eng.PARAMS.max_holding_bars
        max_holding_delta = timedelta(minutes=30 * max_holding_bars)
        if exit_reason is None and (datetime.now(timezone.utc) - entry_time.to_pydatetime()) >= max_holding_delta:
            exit_reason = "max_holding"

        if exit_reason is None:
            print(f"  [{pos['instrument']}] posizione ancora aperta, nessuna condizione di uscita.")
            continue

        close_direction = "SELL" if pos["direction"] == "long" else "BUY"
        print(f"  [{pos['instrument']}] condizione di uscita: {exit_reason} — chiudo (dry_run={DRY_RUN})")
        result = session.close_position(
            deal_id=pos["ig_deal_id"] or "SIMULATA", direction=close_direction,
            size=pos["size"], dry_run=DRY_RUN,
        )

        if pos["direction"] == "long":
            pnl = (current_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["size"]

        d1_query(f"UPDATE live_positions SET status = 'closed' WHERE id = {pos['id']}")
        d1_query(
            "INSERT INTO live_trades (account_type, position_id, instrument, direction, entry_time, "
            "entry_price, exit_time, exit_price, stop_loss, take_profit, atr_at_entry, size, "
            "risk_amount, pnl, exit_reason, causa_esito, rispetto_regole) VALUES ("
            f"'demo', {pos['id']}, '{pos['instrument']}', '{pos['direction']}', '{pos['entry_time']}', "
            f"{pos['entry_price']}, '{datetime.now(timezone.utc).isoformat()}', {current_price}, "
            f"{pos['stop_loss']}, {pos['take_profit']}, {pos.get('atr_at_entry', 'NULL')}, {pos['size']}, "
            f"{pos['risk_amount']}, {pnl}, '{exit_reason}', "
            f"'{'falso segnale' if exit_reason == 'stop_loss' and pnl < 0 else 'NULL'}', 'si')"
        )
        capital += pnl
        d1_query(f"UPDATE live_daily_state SET capital_current = {capital} WHERE trade_date = '{today_str}'")
        print(f"  [{pos['instrument']}] chiusa: pnl={pnl:+.2f} EUR, nuovo capitale investito={capital:.2f} EUR")


def detect_and_open_signals(session: IGSession, today_str: str):
    day_state = get_today_state(today_str)
    if day_state["kill_switch_triggered"]:
        print("Kill switch attivo oggi — nessun nuovo ordine.")
        return
    if day_state["orders_today"] >= eng.PARAMS.max_new_orders_per_day:
        print("Limite ordini/giorno raggiunto — nessun nuovo ordine.")
        return

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    if len(open_positions) >= eng.PARAMS.max_concurrent_positions:
        print("Nessuno slot concorrente libero — nessun nuovo ordine.")
        return
    open_instruments = {p["instrument"] for p in open_positions}

    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    warmup_start = day_start - timedelta(days=WARMUP_DAYS)
    capital = day_state["capital_current"]

    for name, const in SYMBOLS.items():
        if name in open_instruments:
            continue
        inst = eng.INSTRUMENTS[name]
        hist = fetch_historical(const, warmup_start, now + timedelta(minutes=30))
        signals = eng.generate_signals(hist, inst)
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            continue
        last_closed = closed.iloc[-1]
        sig = last_closed["signal"]
        if sig not in ("long", "short"):
            print(f"  [{name}] nessun segnale.")
            continue

        atr = last_closed["atr"]
        if pd.isna(atr):
            print(f"  [{name}] ATR non disponibile, salto.")
            continue

        try:
            price = session.get_price(name)
        except Exception as e:
            print(f"  [{name}] impossibile leggere il prezzo per l'ordine ({e}), salto.")
            continue
        entry_price = price["offer"] if sig == "long" else price["bid"]
        if entry_price is None:
            print(f"  [{name}] prezzo non disponibile, salto.")
            continue

        stop_distance_pts = atr * inst.atr_multiplier
        limit_distance_pts = stop_distance_pts * eng.PARAMS.rr_target
        risk_amount = capital * inst.risk_pct
        size = risk_amount / (stop_distance_pts * inst.point_value)
        forced_min = False
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
            forced_min = True

        direction = "BUY" if sig == "long" else "SELL"
        print(f"  [{name}] segnale {sig.upper()} — size={size:.2f} "
              f"(forzata al minimo={forced_min}) stop={stop_distance_pts:.1f}pt "
              f"target={limit_distance_pts:.1f}pt (dry_run={DRY_RUN})")

        result = session.place_order(
            instrument=name, direction=direction, size=size,
            stop_distance=stop_distance_pts, limit_distance=limit_distance_pts, dry_run=DRY_RUN,
        )

        if DRY_RUN:
            print(f"    Simulato, nessuna scrittura in live_positions (solo in modalità reale).")
            continue

        deal_id = result.get("dealId", "")
        stop_loss = entry_price - stop_distance_pts if sig == "long" else entry_price + stop_distance_pts
        take_profit = entry_price + limit_distance_pts if sig == "long" else entry_price - limit_distance_pts

        d1_query(
            "INSERT INTO live_positions (account_type, instrument, direction, status, entry_time, "
            "entry_price, stop_loss, take_profit, size, risk_amount, atr_at_entry, "
            "max_holding_bars, ig_deal_id) VALUES ("
            f"'demo', '{name}', '{sig}', 'open', '{now.isoformat()}', {entry_price}, "
            f"{stop_loss}, {take_profit}, {size}, {risk_amount}, {atr}, "
            f"{eng.PARAMS.max_holding_bars}, '{deal_id}')"
        )
        d1_query(
            f"UPDATE live_daily_state SET orders_today = orders_today + 1 WHERE trade_date = '{today_str}'"
        )
        print(f"    Posizione aperta su IG, deal_id={deal_id}")


def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN mancanti.")
        sys.exit(1)

    print(f"=== live_execute.py — DRY_RUN={DRY_RUN} — {datetime.now(timezone.utc).isoformat()} ===\n")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        print("--- 1) Gestione posizioni aperte ---")
        manage_open_positions(session, today_str)

        print("\n--- 2) Rilevazione nuovi segnali ---")
        detect_and_open_signals(session, today_str)

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
