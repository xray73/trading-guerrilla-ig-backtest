"""
live_execute.py — Collega check-segnale, sizing (rispettando
l'accantonamento in D1) e ig_client per l'esecuzione su IG demo.

AGGIORNAMENTO 18/07/2026 — INTEGRAZIONE MEAN-REVERSION:
Il ciclo ora gestisce DUE strategie indipendenti, ciascuna col proprio
sotto-pool di capitale (split FISSO 70% V6 / 30% mean-reversion,
deciso su 2.000€ reali — 1.400€/600€ iniziali):

  - V6 (esistente, invariata): segnale eng.generate_signals(), forza
    la size al minimo se sotto soglia (comportamento già validato).
  - Mean-reversion (nuova): segnale generate_mean_reversion_signals()
    variante RSI (selezionata dopo confronto con Bollinger, vedi RCA),
    SALTA il trade se il rischio calcolato non copre la size minima
    (stessa logica di engine_mean_reversion.py, replicata qui perché
    live_execute.py non usa la classe BacktestEngine — calcola la size
    manualmente).

I due pool sono trattati come conti virtuali COMPLETAMENTE indipendenti:
slot concorrenti (max 2) e ordini/giorno (max 3) separati per pool, non
condivisi — stessa filosofia per cui il router combinato (capitale
condiviso) è stato scartato in fase di backtest per il drawdown peggiore.
V6 e mean-reversion POSSONO avere posizioni aperte sullo stesso
strumento contemporaneamente (es. DAX long da V6 + DAX short da MR) —
conseguenza diretta della separazione di capitale, già implicita nei
backtest di split (due motori indipendenti sugli stessi dati).

Accantonamento: rimane un meccanismo UNICO sul capitale COMBINATO
(capital_current_v6 + capital_current_mr) — sul conto IG reale esiste
un solo saldo, i due pool sono solo contabilità interna per il sizing.
Quando scatta il consolidamento mensile, entrambi i pool vengono
ridotti in proporzione al loro peso corrente (non forzati a tornare
70/30 — lo split è fisso solo all'origine).

AGGIORNAMENTO 18/07/2026 (2) — KILL SWITCH GIORNALIERO IMPLEMENTATO:
Fino a questa versione, kill_switch_triggered_v6/mr veniva letto ma
MAI calcolato (gap preesistente, segnalato in una sessione precedente
e ora risolto). Nuova funzione check_and_apply_kill_switches():
SEPARATO per pool (decisione esplicita 18/07/2026) — calcola perdita
realizzata+floating rispetto al capitale di inizio giornata di
ciascun pool; se supera kill_switch_threshold_pct (default -4%),
blocca SOLO nuovi ordini per quel pool per il resto della giornata.
Le posizioni già aperte NON vengono mai chiuse forzatamente — stessa
scelta esplicita già validata in backtest per
BacktestEngineFloatingKillSwitch (una chiusura forzata taglierebbe
anche trade che potrebbero recuperare prima del proprio stop). Chiamata
in main() subito dopo manage_open_positions() e prima della ricerca
di nuovi segnali, così il PnL realizzato del ciclo è già aggiornato.

LIMITE NOTO (identico al backtest): il check avviene una volta per
ciclo (~30 min), non istante per istante — un affondo temporaneo che
rientra prima del prossimo ciclo non viene visto.

DRY_RUN=True di default (variabile d'ambiente DRY_RUN, default "true")
— nessun ordine reale finché non viene impostata esplicitamente a
"false". Invariato dalla versione precedente.

Ciclo ad ogni esecuzione (pensato per girare via cron ogni 30min):
  1. Gestisce le posizioni aperte (V6 + MR indistintamente, la logica
     di chiusura è identica — cambia solo verso quale pool instradare
     il PnL, letto dal campo `strategy` della posizione).
  1b. Verifica kill switch giornaliero, separato per pool.
  2. Rileva nuovi segnali V6 (logica invariata, ora blocca su
     kill_switch_triggered_v6 realmente calcolato).
  3. Rileva nuovi segnali mean-reversion RSI (idem per kill_switch_triggered_mr).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from mean_reversion_signals import generate_mean_reversion_signals
from ig_client import IGSession, load_credentials_from_env

WARMUP_DAYS = 90
CAPITAL0_DEFAULT = 2000.0
SPLIT_V6_PCT = 0.70
SPLIT_MR_PCT = 0.30
MR_MODE = "rsi"  # variante selezionata, vedi RCA Addendum 17-18/07/2026 sez. 45
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


CONSOLIDATE_PCT = 0.4      # opzione 3 mensile, validata su 5 periodi ufficiali il 16-17/07/2026
THRESHOLD_MULT = 1.5


def apply_monthly_consolidation_if_needed(today_str: str, prev_state: dict) -> dict:
    """Accantonamento su capitale COMBINATO (v6+mr) — decisione esplicita
    18/07/2026: sul conto IG reale esiste un solo saldo, i due pool sono
    solo contabilità interna per il sizing. Quando scatta, i due pool
    vengono ridotti in proporzione al loro peso corrente (split fisso
    solo all'origine, non forzato a restare 70/30 nel tempo)."""
    capital_v6 = prev_state.get("capital_current_v6")
    capital_mr = prev_state.get("capital_current_mr")
    if capital_v6 is None or capital_mr is None:
        # riga legacy pre-migrazione: split dal capitale combinato secondo lo split iniziale
        legacy_capital = prev_state["capital_current"]
        capital_v6 = legacy_capital * SPLIT_V6_PCT
        capital_mr = legacy_capital * SPLIT_MR_PCT

    if not prev_state.get("accantonamento_attivo", 1):
        combined = capital_v6 + capital_mr
        return {
            "capital_v6": capital_v6, "capital_mr": capital_mr,
            "accantonato": prev_state.get("accantonato", 0.0) or 0.0,
            "reference": prev_state.get("consolidamento_reference") or combined,
            "threshold": prev_state.get("consolidamento_threshold") or combined * THRESHOLD_MULT,
        }

    accantonato = prev_state.get("accantonato", 0.0) or 0.0
    combined = capital_v6 + capital_mr
    reference = prev_state.get("consolidamento_reference") or combined
    threshold = prev_state.get("consolidamento_threshold") or (reference * THRESHOLD_MULT)

    prev_month = prev_state["trade_date"][:7]
    this_month = today_str[:7]

    if this_month != prev_month:
        while combined > threshold:
            gain = combined - reference
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            reduction_fraction = consolidated / combined
            capital_v6 -= capital_v6 * reduction_fraction
            capital_mr -= capital_mr * reduction_fraction
            accantonato += consolidated
            combined = capital_v6 + capital_mr
            reference = combined
            threshold = reference * THRESHOLD_MULT
            print(f"[accantonamento] Consolidati {consolidated:.2f} EUR al cambio mese "
                  f"({prev_month} -> {this_month}). Investito totale: {combined:.2f} "
                  f"(V6={capital_v6:.2f} MR={capital_mr:.2f})  Accantonato: {accantonato:.2f}")

    return {"capital_v6": capital_v6, "capital_mr": capital_mr,
            "accantonato": accantonato, "reference": reference, "threshold": threshold}


def get_or_create_today_state(today_str: str) -> dict:
    rows = d1_query(f"SELECT * FROM live_daily_state WHERE trade_date = '{today_str}'")
    if rows:
        return rows[0]

    prev_rows = d1_query("SELECT * FROM live_daily_state ORDER BY trade_date DESC LIMIT 1")

    if prev_rows:
        updated = apply_monthly_consolidation_if_needed(today_str, prev_rows[0])
        start_v6, start_mr = updated["capital_v6"], updated["capital_mr"]
        starting_accantonato = updated["accantonato"]
        reference, threshold = updated["reference"], updated["threshold"]
    else:
        start_v6 = CAPITAL0_DEFAULT * SPLIT_V6_PCT
        start_mr = CAPITAL0_DEFAULT * SPLIT_MR_PCT
        starting_accantonato = 0.0
        reference = CAPITAL0_DEFAULT
        threshold = CAPITAL0_DEFAULT * THRESHOLD_MULT

    combined = start_v6 + start_mr
    d1_query(
        "INSERT INTO live_daily_state "
        "(trade_date, account_type, capital_start_of_day, capital_current, "
        "capital_start_of_day_v6, capital_current_v6, capital_start_of_day_mr, capital_current_mr, "
        "accantonato, consolidamento_reference, consolidamento_threshold, accantonamento_attivo) "
        f"VALUES ('{today_str}', 'demo', {combined}, {combined}, "
        f"{start_v6}, {start_v6}, {start_mr}, {start_mr}, "
        f"{starting_accantonato}, {reference}, {threshold}, 1)"
    )
    print(f"Creato nuovo record live_daily_state per {today_str}: "
          f"V6={start_v6:.2f} EUR, MR={start_mr:.2f} EUR, accantonato={starting_accantonato:.2f} EUR")
    return {
        "trade_date": today_str, "account_type": "demo",
        "capital_current_v6": start_v6, "capital_current_mr": start_mr,
        "capital_start_of_day_v6": start_v6, "capital_start_of_day_mr": start_mr,
        "accantonato": starting_accantonato,
        "orders_today_v6": 0, "orders_today_mr": 0,
        "kill_switch_triggered_v6": 0, "kill_switch_triggered_mr": 0,
        "kill_switch_threshold_pct": -4.0,
    }


def get_today_state(today_str: str) -> dict:
    return get_or_create_today_state(today_str)


def manage_open_positions(session: IGSession, today_str: str):
    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    if not open_positions:
        print("Nessuna posizione aperta da gestire.")
        return

    day_state = get_today_state(today_str)
    capital_v6 = day_state["capital_current_v6"]
    capital_mr = day_state["capital_current_mr"]

    for pos in open_positions:
        try:
            price = session.get_price(pos["instrument"])
        except Exception as e:
            print(f"  [{pos['instrument']}/{pos['strategy']}] impossibile leggere il prezzo ({e}), salto.")
            continue

        current_price = price["bid"] if pos["direction"] == "long" else price["offer"]
        if current_price is None:
            print(f"  [{pos['instrument']}/{pos['strategy']}] prezzo non disponibile, salto.")
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
            print(f"  [{pos['instrument']}/{pos['strategy']}] posizione ancora aperta, nessuna condizione di uscita.")
            continue

        close_direction = "SELL" if pos["direction"] == "long" else "BUY"
        print(f"  [{pos['instrument']}/{pos['strategy']}] condizione di uscita: {exit_reason} — chiudo (dry_run={DRY_RUN})")
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
            "risk_amount, pnl, exit_reason, causa_esito, rispetto_regole, strategy) VALUES ("
            f"'demo', {pos['id']}, '{pos['instrument']}', '{pos['direction']}', '{pos['entry_time']}', "
            f"{pos['entry_price']}, '{datetime.now(timezone.utc).isoformat()}', {current_price}, "
            f"{pos['stop_loss']}, {pos['take_profit']}, {pos.get('atr_at_entry', 'NULL')}, {pos['size']}, "
            f"{pos['risk_amount']}, {pnl}, '{exit_reason}', "
            f"'{'falso segnale' if exit_reason == 'stop_loss' and pnl < 0 else 'NULL'}', 'si', '{pos['strategy']}')"
        )

        if pos["strategy"] == "mean_reversion":
            capital_mr += pnl
            d1_query(f"UPDATE live_daily_state SET capital_current_mr = {capital_mr} WHERE trade_date = '{today_str}'")
        else:
            capital_v6 += pnl
            d1_query(f"UPDATE live_daily_state SET capital_current_v6 = {capital_v6} WHERE trade_date = '{today_str}'")

        print(f"  [{pos['instrument']}/{pos['strategy']}] chiusa: pnl={pnl:+.2f} EUR, "
              f"nuovo capitale pool {pos['strategy']}={(capital_mr if pos['strategy']=='mean_reversion' else capital_v6):.2f} EUR")


def check_and_apply_kill_switches(session: IGSession, today_str: str):
    """Kill switch giornaliero, SEPARATO per pool (decisione esplicita
    18/07/2026). Calcola perdita realizzata+floating rispetto al
    capitale di inizio giornata di ciascun pool; se supera la soglia
    (kill_switch_threshold_pct, default -4%), blocca SOLO nuovi ordini
    per quel pool per il resto della giornata — le posizioni già aperte
    restano gestite dai loro stop/target/max_holding normali, MAI
    chiuse forzatamente (stessa scelta esplicita di
    engine_floating_kill_switch.py in backtest: una chiusura forzata
    taglierebbe anche trade che potrebbero recuperare prima del
    proprio stop).

    LIMITE NOTO (identico al backtest): il check avviene una volta per
    ciclo (~30 min), non istante per istante — un affondo temporaneo
    che rientra prima del prossimo ciclo non viene visto.
    """
    day_state = get_today_state(today_str)
    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    threshold_pct = abs(day_state.get("kill_switch_threshold_pct") or -4.0) / 100.0

    instruments_needed = {p["instrument"] for p in open_positions}
    price_cache = {}
    for name in instruments_needed:
        try:
            price_cache[name] = session.get_price(name)
        except Exception as e:
            print(f"  [kill switch] impossibile leggere prezzo {name} ({e}) — "
                  f"posizioni su questo strumento escluse dal calcolo floating questo ciclo.")

    floating_by_strategy = {"v6": 0.0, "mean_reversion": 0.0}
    for pos in open_positions:
        price = price_cache.get(pos["instrument"])
        if price is None:
            continue
        current_price = price["bid"] if pos["direction"] == "long" else price["offer"]
        if current_price is None:
            continue
        if pos["direction"] == "long":
            pnl = (current_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["size"]
        floating_by_strategy[pos["strategy"]] = floating_by_strategy.get(pos["strategy"], 0.0) + pnl

    pools = [
        ("v6", "capital_current_v6", "capital_start_of_day_v6",
         "kill_switch_triggered_v6", "kill_switch_triggered_at_v6", "floating_pnl_today_v6"),
        ("mean_reversion", "capital_current_mr", "capital_start_of_day_mr",
         "kill_switch_triggered_mr", "kill_switch_triggered_at_mr", "floating_pnl_today_mr"),
    ]

    for strategy, capital_field, start_field, flag_field, at_field, floating_field in pools:
        capital_current = day_state.get(capital_field)
        capital_start = day_state.get(start_field)
        if capital_current is None or capital_start is None or capital_start == 0:
            continue

        floating = floating_by_strategy.get(strategy, 0.0)
        d1_query(f"UPDATE live_daily_state SET {floating_field} = {floating} WHERE trade_date = '{today_str}'")

        total_change = (capital_current - capital_start) + floating
        loss_pct = -total_change / capital_start if total_change < 0 else 0.0

        already_triggered = day_state.get(flag_field)
        if not already_triggered and loss_pct >= threshold_pct:
            now_iso = datetime.now(timezone.utc).isoformat()
            d1_query(
                f"UPDATE live_daily_state SET {flag_field} = 1, {at_field} = '{now_iso}' "
                f"WHERE trade_date = '{today_str}'"
            )
            print(f"[kill switch] *** ATTIVATO per {strategy} *** perdita giornaliera "
                  f"{loss_pct*100:.2f}% >= soglia {threshold_pct*100:.2f}% "
                  f"(realizzato+floating vs capitale inizio giornata). Nessun nuovo ordine "
                  f"per {strategy} fino a domani. Posizioni già aperte NON chiuse forzatamente.")
        elif loss_pct > 0:
            print(f"  [kill switch/{strategy}] perdita giornaliera {loss_pct*100:.2f}% "
                  f"(soglia {threshold_pct*100:.2f}%) — {'GIA ATTIVO' if already_triggered else 'sotto soglia'}")


def detect_and_open_signals_v6(session: IGSession, today_str: str, hist_cache: dict):
    day_state = get_today_state(today_str)
    if day_state.get("kill_switch_triggered_v6"):
        print("[V6] Kill switch attivo oggi — nessun nuovo ordine.")
        return
    if (day_state.get("orders_today_v6") or 0) >= eng.PARAMS.max_new_orders_per_day:
        print("[V6] Limite ordini/giorno raggiunto — nessun nuovo ordine.")
        return

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open' AND strategy = 'v6'")
    if len(open_positions) >= eng.PARAMS.max_concurrent_positions:
        print("[V6] Nessuno slot concorrente libero — nessun nuovo ordine.")
        return
    open_instruments = {p["instrument"] for p in open_positions}

    now = datetime.now(timezone.utc)
    capital = day_state["capital_current_v6"]

    for name in SYMBOLS:
        if name in open_instruments:
            continue
        inst = eng.INSTRUMENTS[name]
        hist = hist_cache[name]
        signals = eng.generate_signals(hist, inst)
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            continue
        last_closed = closed.iloc[-1]
        sig = last_closed["signal"]
        if sig not in ("long", "short"):
            print(f"  [V6/{name}] nessun segnale.")
            continue

        atr = last_closed["atr"]
        if pd.isna(atr):
            print(f"  [V6/{name}] ATR non disponibile, salto.")
            continue

        try:
            price = session.get_price(name)
        except Exception as e:
            print(f"  [V6/{name}] impossibile leggere il prezzo per l'ordine ({e}), salto.")
            continue
        entry_price = price["offer"] if sig == "long" else price["bid"]
        if entry_price is None:
            print(f"  [V6/{name}] prezzo non disponibile, salto.")
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
        print(f"  [V6/{name}] segnale {sig.upper()} — size={size:.2f} "
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
            "max_holding_bars, ig_deal_id, strategy) VALUES ("
            f"'demo', '{name}', '{sig}', 'open', '{now.isoformat()}', {entry_price}, "
            f"{stop_loss}, {take_profit}, {size}, {risk_amount}, {atr}, "
            f"{eng.PARAMS.max_holding_bars}, '{deal_id}', 'v6')"
        )
        d1_query(
            f"UPDATE live_daily_state SET orders_today_v6 = orders_today_v6 + 1 WHERE trade_date = '{today_str}'"
        )
        print(f"    Posizione V6 aperta su IG, deal_id={deal_id}")


def detect_and_open_signals_mr(session: IGSession, today_str: str, hist_cache: dict):
    """Identica a detect_and_open_signals_v6 nella struttura, con due
    differenze intenzionali: segnale generate_mean_reversion_signals()
    variante RSI, e size che SALTA (non forza) sotto il minimo —
    replica qui la stessa logica di engine_mean_reversion.py, che non
    può essere riusata direttamente perché live_execute.py non usa la
    classe BacktestEngine per il sizing (calcolo manuale, come V6)."""
    day_state = get_today_state(today_str)
    if day_state.get("kill_switch_triggered_mr"):
        print("[MR] Kill switch attivo oggi — nessun nuovo ordine.")
        return
    if (day_state.get("orders_today_mr") or 0) >= eng.PARAMS.max_new_orders_per_day:
        print("[MR] Limite ordini/giorno raggiunto — nessun nuovo ordine.")
        return

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open' AND strategy = 'mean_reversion'")
    if len(open_positions) >= eng.PARAMS.max_concurrent_positions:
        print("[MR] Nessuno slot concorrente libero — nessun nuovo ordine.")
        return
    open_instruments = {p["instrument"] for p in open_positions}

    now = datetime.now(timezone.utc)
    capital = day_state["capital_current_mr"]

    for name in SYMBOLS:
        if name in open_instruments:
            continue
        inst = eng.INSTRUMENTS[name]
        hist = hist_cache[name]
        signals = generate_mean_reversion_signals(hist, inst, mode=MR_MODE)
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            continue
        last_closed = closed.iloc[-1]
        sig = last_closed["signal"]
        if sig not in ("long", "short"):
            print(f"  [MR/{name}] nessun segnale.")
            continue

        atr = last_closed["atr"]
        if pd.isna(atr):
            print(f"  [MR/{name}] ATR non disponibile, salto.")
            continue

        try:
            price = session.get_price(name)
        except Exception as e:
            print(f"  [MR/{name}] impossibile leggere il prezzo per l'ordine ({e}), salto.")
            continue
        entry_price = price["offer"] if sig == "long" else price["bid"]
        if entry_price is None:
            print(f"  [MR/{name}] prezzo non disponibile, salto.")
            continue

        stop_distance_pts = atr * inst.atr_multiplier
        limit_distance_pts = stop_distance_pts * eng.PARAMS.rr_target
        risk_amount = capital * inst.risk_pct
        size = risk_amount / (stop_distance_pts * inst.point_value)

        if size < inst.min_tradable_size:
            print(f"  [MR/{name}] segnale {sig.upper()} SALTATO — size calcolata {size:.3f} "
                  f"sotto il minimo {inst.min_tradable_size} (capitale pool MR={capital:.2f} EUR "
                  f"insufficiente per questo ATR). Comportamento intenzionale, vedi engine_mean_reversion.py.")
            continue

        direction = "BUY" if sig == "long" else "SELL"
        print(f"  [MR/{name}] segnale {sig.upper()} — size={size:.2f} "
              f"stop={stop_distance_pts:.1f}pt target={limit_distance_pts:.1f}pt (dry_run={DRY_RUN})")

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
            "max_holding_bars, ig_deal_id, strategy) VALUES ("
            f"'demo', '{name}', '{sig}', 'open', '{now.isoformat()}', {entry_price}, "
            f"{stop_loss}, {take_profit}, {size}, {risk_amount}, {atr}, "
            f"{eng.PARAMS.max_holding_bars}, '{deal_id}', 'mean_reversion')"
        )
        d1_query(
            f"UPDATE live_daily_state SET orders_today_mr = orders_today_mr + 1 WHERE trade_date = '{today_str}'"
        )
        print(f"    Posizione MR aperta su IG, deal_id={deal_id}")


def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN mancanti.")
        sys.exit(1)

    print(f"=== live_execute.py — DRY_RUN={DRY_RUN} — {datetime.now(timezone.utc).isoformat()} ===\n")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        print("--- 1) Gestione posizioni aperte (V6 + MR) ---")
        manage_open_positions(session, today_str)

        print("\n--- 1b) Verifica kill switch giornaliero (separato per pool) ---")
        check_and_apply_kill_switches(session, today_str)

        # un solo fetch Dukascopy per strumento, riusato da V6 e MR
        print("\n--- 2) Scarico storico (riusato da entrambe le strategie) ---")
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        warmup_start = day_start - timedelta(days=WARMUP_DAYS)
        hist_cache = {}
        for name, const in SYMBOLS.items():
            hist_cache[name] = fetch_historical(const, warmup_start, now + timedelta(minutes=30))

        print("\n--- 3) Rilevazione nuovi segnali V6 ---")
        detect_and_open_signals_v6(session, today_str, hist_cache)

        print("\n--- 4) Rilevazione nuovi segnali mean-reversion (RSI) ---")
        detect_and_open_signals_mr(session, today_str, hist_cache)

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
