"""
check_margin_conflict.py — Screening: quante volte, sui 5 periodi ufficiali
storici (11 anni, DAX+FTSE100), un controllo di margine reale AGGREGATO
(V6+MR sullo stesso conto/equity condivisa) avrebbe bloccato un trade che
il backtest attuale (due motori separati, mai margine condiviso) ha
eseguito senza problemi.

APPROSSIMAZIONI DICHIARATE (nessuna nascosta):
1. entry_price approssimato con close_entry (non l'open della barra
   successiva come nel motore reale) — stesso livello di approssimazione
   gia' dichiarato in altri script di questo tipo (bootstrap_gold).
2. Equity di riferimento FISSA a 2.000 EUR (capitale reale nominale),
   non equity che si evolve col PnL — semplificazione per screening,
   coerente con l'uso di capital0 fisso per periodo nel resto del progetto.
3. Uscita (stop/target/max_holding) ricostruita rigiocando i 49 bar di
   research_v6/mr_candidate_path — stessa logica del motore reale (ATR x
   1.5 stop, RR 2 target, max 48 barre), NON rilegge trades.pnl (non
   affidabile per il join con MR, mai loggato in backtest_runs).
4. Alla parita' di timestamp, le CHIUSURE vengono processate prima delle
   APERTURE (libera margine prima di verificarne la disponibilita' per un
   nuovo trade) — assunzione conservativa, favorisce il motore.

Output: SOLO aggregati (conteggio blocchi, PnL approssimato perso/pool),
nessun dato individuale scaricato in chat.
"""
import requests
import os
from datetime import datetime, timedelta

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

MARGIN_PCT = 0.05
POINT_VALUE = 1.0
ATR_MULT = 1.5
RR_TARGET = 2.0
MAX_HOLDING_BARS = 48
EQUITY_REF = 2000.0
RISK_PCT = {"DAX": 0.02, "FTSE100": 0.015}
MIN_SIZE = 0.50
POOL_CAPITAL = {"v6": 1400.0, "mr": 600.0}


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]["results"]


def fetch_executed(table):
    return d1(f"SELECT candidate_key, instrument, direction, entry_time, "
              f"atr_at_entry, close_entry FROM {table} WHERE was_executed=1")


def fetch_path(path_table, candidate_key):
    rows = d1(f"SELECT bar_offset, high, low, close FROM {path_table} "
              f"WHERE candidate_key = '{candidate_key}' ORDER BY bar_offset ASC")
    return rows


def simulate_exit(cand, path_table):
    """Rigioca la path bar-per-bar, ritorna (exit_offset, r_multiple_approx)."""
    entry_price = cand["close_entry"]
    atr = cand["atr_at_entry"]
    if entry_price is None or atr is None:
        return None, None
    stop_dist = atr * ATR_MULT
    target_dist = stop_dist * RR_TARGET
    direction = cand["direction"]

    if direction == "long":
        stop_price = entry_price - stop_dist
        target_price = entry_price + target_dist
    else:
        stop_price = entry_price + stop_dist
        target_price = entry_price - target_dist

    path = fetch_path(path_table, cand["candidate_key"])
    if not path:
        return None, None

    for bar in path:
        offset = bar["bar_offset"]
        if offset == 0:
            continue  # barra di entrata, nessun check
        high, low, close = bar["high"], bar["low"], bar["close"]
        if high is None or low is None:
            continue
        if direction == "long":
            if low <= stop_price:
                return offset, -1.0
            if high >= target_price:
                return offset, RR_TARGET
        else:
            if high >= stop_price:
                return offset, -1.0
            if low <= target_price:
                return offset, RR_TARGET
        if offset >= MAX_HOLDING_BARS:
            r = (close - entry_price) / stop_dist if direction == "long" else (entry_price - close) / stop_dist
            return offset, r

    last = path[-1]
    r = ((last["close"] - entry_price) / stop_dist if direction == "long"
         else (entry_price - last["close"]) / stop_dist)
    return last["bar_offset"], r


def build_events(strategy, candidates, path_table):
    events = []
    for cand in candidates:
        exit_offset, r_mult = simulate_exit(cand, path_table)
        if exit_offset is None:
            continue
        entry_time = datetime.fromisoformat(cand["entry_time"])
        exit_time = entry_time + timedelta(minutes=30 * exit_offset)

        inst = cand["instrument"]
        entry_price = cand["close_entry"]
        atr = cand["atr_at_entry"]
        stop_dist = atr * ATR_MULT
        risk_pct = RISK_PCT[inst]
        capital = POOL_CAPITAL[strategy]
        risk_amount = capital * risk_pct
        size = risk_amount / (stop_dist * POINT_VALUE)
        if size < MIN_SIZE:
            size = MIN_SIZE  # trade fu eseguito -> assume size minima applicata
        margin_required = size * entry_price * POINT_VALUE * MARGIN_PCT
        pnl_approx = r_mult * risk_amount

        events.append({"time": entry_time, "type": "open", "strategy": strategy,
                        "margin": margin_required, "candidate_key": cand["candidate_key"],
                        "pnl_approx": pnl_approx})
        events.append({"time": exit_time, "type": "close", "strategy": strategy,
                        "margin": margin_required, "candidate_key": cand["candidate_key"]})
    return events


def main():
    print("Scarico candidati eseguiti V6...")
    v6_cands = fetch_executed("research_v6_candidates")
    print(f"  {len(v6_cands)} candidati V6 eseguiti")

    print("Scarico candidati eseguiti MR...")
    mr_cands = fetch_executed("research_mr_candidates")
    print(f"  {len(mr_cands)} candidati MR eseguiti")

    print("Ricostruisco uscite V6 (rigioco path)...")
    v6_events = build_events("v6", v6_cands, "research_v6_candidate_path")
    print(f"  {len(v6_events)//2} trade V6 con uscita ricostruita")

    print("Ricostruisco uscite MR (rigioco path)...")
    mr_events = build_events("mr", mr_cands, "research_mr_candidate_path")
    print(f"  {len(mr_events)//2} trade MR con uscita ricostruita")

    all_events = v6_events + mr_events
    # chiusure prima delle aperture a parita' di timestamp (conservativo)
    all_events.sort(key=lambda e: (e["time"], 0 if e["type"] == "close" else 1))

    margin_used = 0.0
    open_margins = {}  # candidate_key -> margin
    blocked = []
    executed_ok = 0

    for ev in all_events:
        if ev["type"] == "close":
            m = open_margins.pop(ev["candidate_key"], None)
            if m is not None:
                margin_used -= m
        else:
            margine_libero = EQUITY_REF - margin_used
            if ev["margin"] > margine_libero:
                blocked.append(ev)
            else:
                margin_used += ev["margin"]
                open_margins[ev["candidate_key"]] = ev["margin"]
                executed_ok += 1

    n_total = executed_ok + len(blocked)
    pnl_blocked = sum(e["pnl_approx"] for e in blocked)
    pnl_blocked_v6 = sum(e["pnl_approx"] for e in blocked if e["strategy"] == "v6")
    pnl_blocked_mr = sum(e["pnl_approx"] for e in blocked if e["strategy"] == "mr")

    print("\n=== RISULTATO AGGREGATO ===")
    print(f"Trade totali analizzati (V6+MR, was_executed=1, 11 anni): {n_total}")
    print(f"Trade che il margine aggregato AVREBBE bloccato: {len(blocked)} "
          f"({100*len(blocked)/n_total:.2f}%)")
    print(f"  di cui V6: {sum(1 for e in blocked if e['strategy']=='v6')}")
    print(f"  di cui MR: {sum(1 for e in blocked if e['strategy']=='mr')}")
    print(f"PnL approssimato (R-multiple x risk_amount) dei trade bloccati: "
          f"{pnl_blocked:+.2f} EUR")
    print(f"  di cui V6: {pnl_blocked_v6:+.2f} EUR")
    print(f"  di cui MR: {pnl_blocked_mr:+.2f} EUR")

    # traccia il picco margine realmente impegnato (simulazione con blocco attivo)
    margin_used2 = 0.0
    open_margins2 = {}
    peak = 0.0
    for ev in all_events:
        if ev["type"] == "close":
            m = open_margins2.pop(ev["candidate_key"], None)
            if m is not None:
                margin_used2 -= m
        else:
            margine_libero = EQUITY_REF - margin_used2
            if ev["margin"] <= margine_libero:
                margin_used2 += ev["margin"]
                open_margins2[ev["candidate_key"]] = ev["margin"]
                peak = max(peak, margin_used2)
    print(f"Margine di picco realmente impegnato (con blocco attivo): {peak:.2f} EUR "
          f"({100*peak/EQUITY_REF:.1f}% dell'equity di riferimento {EQUITY_REF:.0f} EUR)")


if __name__ == "__main__":
    main()
