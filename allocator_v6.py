"""
allocator_v6.py — Allocatore leggero fedele al motore di produzione
(BacktestEngineFloatingKillSwitch), costruito sopra i dati gia' estratti
in research_v6_candidates / research_v6_candidate_path invece di
rigirare l'intero motore sui prezzi grezzi.

Perche' serve (principio 9, Protocollo Anti-Rumore): i trade non sono
indipendenti — quali segnali diventano trade dipende da quali slot sono
liberi in quel momento, che dipende a sua volta dalla regola di uscita
in vigore. Cambiare la regola di uscita puo' far entrare candidati che
nella storia reale non sono mai stati eseguiti (slot occupato).

DUE FASI, DELIBERATAMENTE SEPARATE:

FASE 1 — MASCHERAMENTO (mask_candidate): puramente meccanico, usa SOLO
i prezzi (OHLC del path 49 barre, neutro, gia' non cappato dal fix
offset-48 del 21/07/2026). Nessuna dipendenza da capitale/slot/stato.
Replica ESATTAMENTE _try_close_position()/_open_position() di engine.py:
- entry_price = open barra offset 0 +/- spread/2
- stop/target controllati nell'ordine stop-poi-target su high/low,
  offset 1..48
- se nessuno dei due scatta e offset==48 (bars_held>=max_holding_bars),
  uscita a mercato al close di quella barra +/- spread/2
- se il path e' troppo corto (candidato troppo recente, manca offset
  48) il candidato viene marcato incompleto e ESCLUSO dalla simulazione
  (nota nel riepilogo finale, non silenzioso)

FASE 2 — ALLOCAZIONE SEQUENZIALE (Allocator): simula bar-by-bar
l'ordine cronologico reale, replicando esattamente
engine_floating_kill_switch.py: reset giornaliero, floating kill switch
controllato OGNI barra (non solo a chiusura trade) sulle posizioni
aperte, priorita' R:R poi correlazione, mai 2 posizioni sullo stesso
strumento, max 2 concorrenti, max 3 ordini/giorno.

SANITY CHECK OBBLIGATORIO: con TEST_SCOPE che isola un solo periodo/
strumento, il numero di trade e il PnL totale devono combaciare con
research_v6_trade_features per lo stesso filtro — verificato a fine
run, stampato nel log, MAI dato per scontato.

Uso: TEST_SCOPE=dax_2015_2016 per il primo test ridotto (come da
protocollo concordato in chat 21/07/2026 — verificare in piccolo prima
di girare su tutto). TEST_SCOPE=full per la corsa completa V6.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
import requests
import pandas as pd

import engine as eng

D1_DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
D1_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

TEST_SCOPE = os.environ.get("TEST_SCOPE", "dax_2015_2016").strip().lower()


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{D1_API_BASE}/{account_id}/d1/database/{D1_DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Query D1 fallita: {data.get('errors')}")
    return data["result"][0]["results"]


# =====================================================================
# FASE 1 — MASCHERAMENTO (nessuno stato, solo prezzi)
# =====================================================================

@dataclass
class MaskedOutcome:
    candidate_key: str
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_bar_offset: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    atr_at_entry: float
    adx_at_entry: float
    complete: bool


def mask_candidate(candidate_key: str, instrument: str, direction: str,
                    atr_at_entry: float, adx_at_entry: float,
                    path_rows: list[dict], p: eng.ChartaParams,
                    inst: eng.InstrumentConfig) -> MaskedOutcome:
    """path_rows: righe bar_offset 0..48 ordinate, con open/high/low/close/timestamp."""
    if not path_rows or path_rows[0]["bar_offset"] != 0:
        return MaskedOutcome(candidate_key, instrument, direction, None, None,
                              None, None, None, None, None, "incompleto",
                              atr_at_entry, adx_at_entry, False)

    spread = inst.spread_fixed
    entry_bar = path_rows[0]
    raw_open = entry_bar["open"]
    entry_time = pd.Timestamp(entry_bar["timestamp"])
    entry_price = raw_open + spread / 2 if direction == "long" else raw_open - spread / 2

    stop_distance = atr_at_entry * inst.atr_multiplier
    if direction == "long":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + stop_distance * p.rr_target
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - stop_distance * p.rr_target

    by_offset = {r["bar_offset"]: r for r in path_rows}
    for offset in range(1, p.max_holding_bars + 1):
        row = by_offset.get(offset)
        if row is None:
            return MaskedOutcome(candidate_key, instrument, direction, entry_time,
                                  entry_price, stop_loss, take_profit, None, None, None,
                                  "incompleto", atr_at_entry, adx_at_entry, False)
        high, low, close = row["high"], row["low"], row["close"]
        exit_reason, exit_price = None, None

        if direction == "long":
            if low <= stop_loss:
                exit_reason, exit_price = "stop_loss", stop_loss - spread / 2
            elif high >= take_profit:
                exit_reason, exit_price = "take_profit", take_profit - spread / 2
        else:
            if high >= stop_loss:
                exit_reason, exit_price = "stop_loss", stop_loss + spread / 2
            elif low <= take_profit:
                exit_reason, exit_price = "take_profit", take_profit + spread / 2

        if exit_reason is None and offset >= p.max_holding_bars:
            exit_reason = "max_holding"
            exit_price = close - spread / 2 if direction == "long" else close + spread / 2

        if exit_reason:
            return MaskedOutcome(candidate_key, instrument, direction, entry_time,
                                  entry_price, stop_loss, take_profit, offset,
                                  pd.Timestamp(row["timestamp"]), exit_price, exit_reason,
                                  atr_at_entry, adx_at_entry, True)

    return MaskedOutcome(candidate_key, instrument, direction, entry_time, entry_price,
                          stop_loss, take_profit, None, None, None, "incompleto",
                          atr_at_entry, adx_at_entry, False)


# =====================================================================
# FASE 2 — ALLOCAZIONE SEQUENZIALE (replica engine_floating_kill_switch.py)
# =====================================================================

@dataclass
class OpenPos:
    candidate_key: str
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    risk_amount: float
    masked: MaskedOutcome


class Allocator:
    def __init__(self, capital0: float, p: eng.ChartaParams, instruments: dict):
        self.capital0 = capital0
        self.capital = capital0
        self.p = p
        self.instruments = instruments
        self.open_positions: list[OpenPos] = []
        self.closed_trades: list[dict] = []
        self._day_start_capital = capital0
        self._current_day = None
        self._orders_today = 0
        self._kill_switch_active = False

    def _reset_day_if_needed(self, ts: pd.Timestamp):
        day = ts.date()
        if self._current_day != day:
            self._current_day = day
            self._day_start_capital = self.capital
            self._orders_today = 0
            self._kill_switch_active = False

    def _position_size(self, entry_price: float, stop_price: float, inst: eng.InstrumentConfig):
        risk_amount = self.capital * inst.risk_pct
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0
        size = risk_amount / (risk_distance * inst.point_value)
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        if margin_required > self.capital:
            max_size = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            size = max(max_size, 0.0)
        return size, risk_amount

    def _correlation_penalty(self, candidate_instrument: str) -> int:
        penalty = 0
        for pos in self.open_positions:
            penalty += 2 if pos.instrument == candidate_instrument else 1
        return penalty

    def _bar_price_at(self, pos: OpenPos, ts: pd.Timestamp, path_by_key: dict):
        """Cerca nel path GIA' CARICATO del candidato la barra corrispondente
        a ts (per il floating PnL) — nessun fetch aggiuntivo."""
        rows = path_by_key.get(pos.candidate_key)
        if not rows:
            return None
        for r in rows:
            if pd.Timestamp(r["timestamp"]) == ts:
                return r["close"]
        return None

    def _floating_loss_pct(self, ts: pd.Timestamp, path_by_key: dict) -> float:
        floating_pnl = 0.0
        for pos in self.open_positions:
            close_price = self._bar_price_at(pos, ts, path_by_key)
            if close_price is None:
                continue
            if pos.direction == "long":
                floating_pnl += (close_price - pos.entry_price) * pos.size
            else:
                floating_pnl += (pos.entry_price - close_price) * pos.size
        realized_change = self.capital - self._day_start_capital
        total_change = realized_change + floating_pnl
        if self._day_start_capital == 0:
            return 0.0
        pct = total_change / self._day_start_capital
        return abs(pct) if pct < 0 else 0.0

    def _close(self, pos: OpenPos, exit_time, exit_price: float, exit_reason: str):
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
        r_multiple = pnl / pos.risk_amount if pos.risk_amount else 0.0
        self.capital += pnl
        self.closed_trades.append({
            "candidate_key": pos.candidate_key, "instrument": pos.instrument,
            "direction": pos.direction, "entry_time": pos.entry_time,
            "entry_price": pos.entry_price, "exit_time": exit_time,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "size": pos.size, "risk_amount": pos.risk_amount, "pnl": pnl,
            "r_multiple": r_multiple,
        })
        self.open_positions.remove(pos)
        if self._daily_pnl_pct() <= -self.p.kill_switch_pct:
            self._kill_switch_active = True

    def _daily_pnl_pct(self) -> float:
        if self._day_start_capital == 0:
            return 0.0
        return (self.capital - self._day_start_capital) / self._day_start_capital

    def run(self, masked: list[MaskedOutcome], path_by_key: dict):
        """masked: SOLO candidati con complete=True, ordinati per entry_time.
        path_by_key: candidate_key -> lista righe path (per floating PnL)."""
        # timeline: ogni barra rilevante = entry_time di un candidato UNION
        # ogni timestamp presente nel path di un candidato aperto durante
        # la simulazione (per il floating check ad ogni barra)
        by_entry = {}
        for m in masked:
            by_entry.setdefault(m.entry_time, []).append(m)

        all_ts = sorted(set(
            pd.Timestamp(r["timestamp"])
            for rows in path_by_key.values() for r in rows
        ))

        for ts in all_ts:
            self._reset_day_if_needed(ts)

            for pos in list(self.open_positions):
                if pos.masked.exit_time == ts:
                    self._close(pos, pos.masked.exit_time, pos.masked.exit_price,
                                pos.masked.exit_reason)

            if not self._kill_switch_active and self.open_positions:
                perdita_pct = self._floating_loss_pct(ts, path_by_key)
                if perdita_pct >= self.p.kill_switch_pct:
                    self._kill_switch_active = True

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            candidates_here = by_entry.get(ts, [])
            if not candidates_here:
                continue
            candidates_here = [c for c in candidates_here
                                if not any(p.instrument == c.instrument for p in self.open_positions)]
            if not candidates_here:
                continue

            candidates_here.sort(key=lambda c: self._correlation_penalty(c.instrument))

            slots_free = self.p.max_concurrent_positions - len(self.open_positions)
            for c in candidates_here:
                if slots_free <= 0 or self._orders_today >= self.p.max_new_orders_per_day:
                    break
                inst = self.instruments[c.instrument]
                size, risk_amount = self._position_size(c.entry_price, c.stop_loss, inst)
                if size <= 0:
                    continue
                self.open_positions.append(OpenPos(
                    c.candidate_key, c.instrument, c.direction, c.entry_time,
                    c.entry_price, size, risk_amount, c))
                self._orders_today += 1
                slots_free -= 1

        return pd.DataFrame(self.closed_trades)


# =====================================================================
# CARICAMENTO DATI + SCOPE DI TEST
# =====================================================================

SCOPES = {
    # ATTENZIONE: DAX isolato NON e' confrontabile 1:1 con research_v6_trade_features
    # (che include sempre la competizione di slot con FTSE100) — utile solo come
    # diagnostica strutturale, MAI come sanity check di conteggio trade.
    "dax_2015_2016": {"instruments": ["DAX"], "start": "2015-01-05", "end": "2017-01-01"},
    # Scope corretto per il sanity check: entrambi gli strumenti, stessa
    # competizione di slot della storia reale — 2015-2016 come primo periodo.
    "dax_ftse_2015_2016": {"instruments": ["DAX", "FTSE100"], "start": "2015-01-05", "end": "2017-01-01"},
    "full": {"instruments": ["DAX", "FTSE100"], "start": "2015-01-05", "end": "2027-01-01"},
}

CAPITAL0_BY_SCOPE = {
    # 2.000 EUR verificato dal risk_amount del primo trade reale (40 = 2000*2%)
    # per ciascuno dei 5 periodi ufficiali (reset a inizio periodo).
    "dax_2015_2016": 2000.0,
    "dax_ftse_2015_2016": 2000.0,
    "full": 2000.0,
}


def load_candidates_and_paths(account_id: str, token: str, scope: dict):
    inst_list = "', '".join(scope["instruments"])
    cands = d1_query(
        f"SELECT candidate_key, instrument, direction, entry_time, atr_at_entry, adx_at_entry "
        f"FROM research_v6_candidates WHERE instrument IN ('{inst_list}') "
        f"AND entry_time >= '{scope['start']}' AND entry_time < '{scope['end']}' "
        f"ORDER BY entry_time",
        account_id, token)

    path_by_key: dict[str, list[dict]] = {}
    keys = [c["candidate_key"] for c in cands]
    CHUNK = 40
    for i in range(0, len(keys), CHUNK):
        chunk_keys = keys[i:i + CHUNK]
        keys_sql = "', '".join(k.replace("'", "''") for k in chunk_keys)
        rows = d1_query(
            f"SELECT candidate_key, bar_offset, timestamp, open, high, low, close "
            f"FROM research_v6_candidate_path WHERE candidate_key IN ('{keys_sql}') "
            f"ORDER BY candidate_key, bar_offset",
            account_id, token)
        for r in rows:
            path_by_key.setdefault(r["candidate_key"], []).append(r)

    return cands, path_by_key


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: credenziali D1 mancanti.")
        sys.exit(1)

    scope = SCOPES.get(TEST_SCOPE)
    if scope is None:
        print(f"ERRORE: TEST_SCOPE '{TEST_SCOPE}' non riconosciuto. Validi: {list(SCOPES)}")
        sys.exit(1)

    print(f"=== Allocatore V6 — TEST_SCOPE={TEST_SCOPE} — {datetime.now(timezone.utc).isoformat()} ===\n")
    print(f"Strumenti: {scope['instruments']}, periodo: {scope['start']} -> {scope['end']}\n")

    print("Carico candidati + path da D1...")
    cands, path_by_key = load_candidates_and_paths(account_id, token, scope)
    print(f"  {len(cands)} candidati caricati, {len(path_by_key)} con path disponibile.\n")

    print("Fase 1 — Mascheramento (solo prezzi, nessuno stato)...")
    masked_all = []
    n_incomplete = 0
    for c in cands:
        rows = path_by_key.get(c["candidate_key"], [])
        rows_sorted = sorted(rows, key=lambda r: r["bar_offset"])
        inst = eng.INSTRUMENTS[c["instrument"]]
        m = mask_candidate(c["candidate_key"], c["instrument"], c["direction"],
                            c["atr_at_entry"], c["adx_at_entry"], rows_sorted,
                            eng.PARAMS, inst)
        if not m.complete:
            n_incomplete += 1
            continue
        masked_all.append(m)
    print(f"  {len(masked_all)} candidati mascherati con successo, {n_incomplete} incompleti (esclusi).\n")

    masked_all.sort(key=lambda m: m.entry_time)

    print("Fase 2 — Allocazione sequenziale (floating kill switch bar-by-bar)...")
    capital0 = CAPITAL0_BY_SCOPE[TEST_SCOPE]
    allocator = Allocator(capital0=capital0, p=eng.PARAMS, instruments=eng.INSTRUMENTS)
    trades_df = allocator.run(masked_all, path_by_key)
    print(f"  {len(trades_df)} trade generati dall'allocatore.")
    if not trades_df.empty:
        print(f"  PnL totale: {trades_df['pnl'].sum():+.2f}")
        print(f"  Distribuzione exit_reason:\n{trades_df['exit_reason'].value_counts().to_string()}")

    if TEST_SCOPE == "dax_2015_2016":
        print("\n=== NOTA: scope DAX isolato — NON comparabile 1:1 con research_v6_trade_features ===")
        print("  (manca la competizione di slot con FTSE100 presente nella storia reale — "
              "usare 'dax_ftse_2015_2016' per il sanity check vero).")
    elif TEST_SCOPE in ("dax_ftse_2015_2016", "full"):
        label = "2015-2016" if TEST_SCOPE == "dax_ftse_2015_2016" else "tutto lo storico"
        print(f"\n=== SANITY CHECK contro research_v6_trade_features (DAX+FTSE100, {label}) ===")
        inst_list = "', '".join(scope["instruments"])
        ref = d1_query(
            f"SELECT COUNT(*) as n, ROUND(SUM(pnl),2) as pnl_tot FROM research_v6_trade_features "
            f"WHERE instrument IN ('{inst_list}') AND entry_time >= '{scope['start']}' "
            f"AND entry_time < '{scope['end']}'",
            account_id, token)
        n_ref = ref[0]["n"]
        pnl_ref = ref[0]["pnl_tot"]
        n_alloc = len(trades_df)
        pnl_alloc = round(trades_df["pnl"].sum(), 2) if not trades_df.empty else 0.0
        print(f"  Riferimento (research_v6_trade_features): {n_ref} trade, PnL={pnl_ref}")
        print(f"  Allocatore (questo run):                  {n_alloc} trade, PnL={pnl_alloc}")
        if n_ref == n_alloc:
            print("  ESITO: numero trade COMBACIA.")
        else:
            print(f"  ESITO: DISCREPANZA di {n_alloc - n_ref} trade — da investigare prima di procedere oltre.")

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
