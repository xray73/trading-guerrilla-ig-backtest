"""
live_execute.py — Collega check-segnale, sizing (rispettando
l'accantonamento in D1) e ig_client per l'esecuzione su IG demo.

AGGIORNAMENTO 21/07/2026 (parte 3) — LIVELLO 3: MARGINE REALE AGGREGATO:
Il tetto equity (Livello 1) confronta solo il capitale di RISCHIO
tracciato (capital_v6+capital_mr) contro l'equity reale — non il
MARGINE effettivamente bloccato da IG per le posizioni aperte in
questo istante. Gap scoperto in chat: capital_v6/capital_mr non si
riducono mai all'apertura di una posizione (solo alla chiusura, col
PnL), quindi il motore poteva in teoria calcolare size "corrette" per
il pool ma per cui IG non aveva più margine libero reale (specialmente
con fino a 4 posizioni concorrenti possibili: V6 max 2 + MR max 2,
condividono lo stesso conto/equity reale ma i loro pool sono contabili
separati). Stesso gap esisteva già nel backtest offline (engine.py,
_position_size: controllo margine c'è ma confronta solo contro
self.capital totale del pool, mai contro margine già impegnato da
altre posizioni concorrenti nello stesso pool — mai testato il caso
V6+MR condividenti lo stesso conto).

Nuova funzione compute_margin_state(): legge equity reale IG + somma
il margine di TUTTE le posizioni aperte (V6+MR, prezzo corrente) →
margine_libero. Prima di ogni nuovo ordine (in detect_and_open_signals_
v6/mr, dopo il sizing finale) si verifica che margin_required non superi
margine_libero — se insufficiente, skip_reason="margine_insufficiente"
(niente "forza al minimo" qui: forzare la size aumenterebbe il margine
richiesto, l'opposto di quanto serve). margine_libero si aggiorna
dentro lo stesso ciclo quando un ordine viene REALMENTE piazzato
(non-DRY_RUN), cosi' un secondo segnale nello stesso giro (es. V6 poi
MR) vede il margine gia' ridotto. Fail-safe: se equity non leggibile,
warning e nessun blocco (stesso pattern gia' usato in apply_equity_cap).

AGGIORNAMENTO 21/07/2026 (parte 2) — TRACKING CANDIDATI LIVE:
Estende detect_and_open_signals_v6/mr per registrare OGNI segnale
valido generato (eseguito o no) in research_v6/mr_candidates +
live_v6/mr_candidates_tracking (stesso formato candidate_key di
extract_v6/mr_candidates.py: f"{instrument}_{entry_time.isoformat()}",
entry_time = barra successiva alla barra di segnale). Rimosso lo skip
anticipato "if name in open_instruments: continue" — il segnale viene
ORA sempre calcolato per ogni strumento, poi si decide separatamente
se aprire l'ordine o solo tracciare il candidato (skip_reason
esplicito: kill_switch_attivo, limite_ordini_giorno, slot_pieno,
strumento_gia_in_posizione, atr_non_disponibile,
prezzo_non_disponibile, size_sotto_minimo_dopo_valvola [solo MR],
margine_insufficiente, dry_run). Nuova funzione log_candidate_bars()
logga ad ogni ciclo la prossima barra disponibile per ogni candidato
ancora in tracking, in live_v6/mr_candidate_bars (completo quando
esiste bar_offset=48, nessun flag is_complete separato — stesso
principio di live_position_bars -> live_closed_position_bars).
Tabelle live_v6/mr_candidates_tracking e live_v6/mr_candidate_bars gia'
create manualmente su D1 (IF NOT EXISTS, non da questo script),
comprese di colonna skip_reason aggiunta il 21/07/2026.

AGGIORNAMENTO 21/07/2026 (parte 1) — LOG BAR-PER-BAR POSIZIONI APERTE:
Aggiunto `live_position_bars` (aggiornata OGNI ciclo mentre una
posizione resta aperta) + `live_closed_position_bars` (archivio
permanente, popolato e la riga viva cancellata alla chiusura — evita
bloat del D1 free tier come da specifica concordata in chat 21/07/2026).
Tabelle create manualmente su D1 (IF NOT EXISTS, non da questo script).
Nessuna modifica alla logica di trading esistente (segnali, sizing,
kill switch, accantonamento) — solo aggiunta di due funzioni di
logging e il loro punto di chiamata dentro manage_open_positions().
Riordinato lo scarico storico (hist_cache) PRIMA della gestione
posizioni aperte (era dopo) perché il logging bar-per-bar ha bisogno
degli indicatori calcolati sull'ultima barra chiusa.

AGGIORNAMENTO 20/07/2026 — DIAGNOSTICA PARAMETRI OGNI CICLO:
Dopo 2 giorni quasi completi senza nessun segnale (ne' V6 ne' MR),
richiesta di visibilita' sui valori reali dei parametri che
determinano il segnale, per verificare che sia plausibile e non un
bug silenzioso (stesso spirito del bug OHLC insert trovato in
sessione — mai fidarsi ciecamente di un "nessun segnale" senza poter
vedere i numeri dietro). Aggiunte le funzioni _diag_v6() e _diag_mr(),
chiamate ad OGNI ciclo (segnale trovato o no) — nessuna modifica alla
logica di generate_signals()/generate_mean_reversion_signals() in
engine.py/mean_reversion_signals.py, solo lettura e stampa dei valori
gia' calcolati sull'ultima barra chiusa.

Il resto del file (split V6/MR, kill switch floating, accantonamento
mensile, protezione accantonato Livello 1/2) e' INVARIATO — vedi
commenti storici sotto per il contesto.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from mean_reversion_signals import generate_mean_reversion_signals, ADX_THRESHOLD, RSI_OVERSOLD, RSI_OVERBOUGHT
from ig_client import IGSession, load_credentials_from_env

WARMUP_DAYS = 90
CAPITAL0_DEFAULT = 2000.0
SPLIT_V6_PCT = 0.70
SPLIT_MR_PCT = 0.30
MR_MODE = "rsi"  # variante selezionata, vedi RCA Addendum 17-18/07/2026 sez. 45
VALVOLA_PCT = 0.20  # scelto 19/07/2026 dopo simulazione sintetica dedicata, vedi docstring
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


# =====================================================================
# DIAGNOSTICA (20/07/2026) — sola lettura, nessun impatto sul segnale
# =====================================================================

def _diag_v6(name: str, row: pd.Series) -> str:
    """Stampa i valori reali dei 4 sub-criteri V6 sull'ultima barra
    chiusa, indipendentemente dal fatto che il segnale scatti o meno.
    Legge SOLO colonne gia' calcolate da eng.generate_signals() —
    nessun ricalcolo, nessuna logica duplicata."""
    close = row["close"]
    ema_f, ema_s = row["ema_fast"], row["ema_slow"]
    adx = row["adx"]
    rh, rl = row["rolling_high"], row["rolling_low"]
    ebf, ebs = row["ema_broad_fast"], row["ema_broad_slow"]

    direction = "long" if ema_f > ema_s else ("short" if ema_f < ema_s else "flat")
    adx_ok = adx > eng.PARAMS.adx_min_context
    dist_breakout_long = close - rh if pd.notna(rh) else float("nan")
    dist_breakout_short = rl - close if pd.notna(rl) else float("nan")
    broad = "long_ok" if ebf > ebs else ("short_ok" if ebf < ebs else "flat")

    return (
        f"  [V6/{name}] diag: close={close:.1f} | ema20={ema_f:.1f} ema50={ema_s:.1f} "
        f"-> direzione={direction} | adx={adx:.1f} (soglia>{eng.PARAMS.adx_min_context:.0f}: "
        f"{'OK' if adx_ok else 'NO'}) | breakout: high{eng.INSTRUMENTS[name].breakout_lookback}="
        f"{rh:.1f} low{eng.INSTRUMENTS[name].breakout_lookback}={rl:.1f} "
        f"(dist da long={dist_breakout_long:+.1f}pt, dist da short={dist_breakout_short:+.1f}pt, "
        f"serve >0 per scattare) | trend ampio: ema100={ebf:.1f} ema200={ebs:.1f} -> {broad}"
    )


def _diag_mr(name: str, row: pd.Series, mode: str) -> str:
    """Analogo a _diag_v6 ma per il ramo mean-reversion. Legge SOLO
    colonne gia' calcolate da generate_mean_reversion_signals()."""
    adx = row["adx"]
    regime_ok = adx < ADX_THRESHOLD
    base = (f"  [MR/{name}] diag: adx={adx:.1f} (regime<{ADX_THRESHOLD:.0f}: "
            f"{'OK' if regime_ok else 'NO, V6 ha priorita in questo regime'})")
    if mode == "rsi" and "rsi" in row.index:
        rsi = row["rsi"]
        base += (f" | rsi={rsi:.1f} (long se <{RSI_OVERSOLD:.0f}, "
                  f"short se >{RSI_OVERBOUGHT:.0f})")
    return base


# =====================================================================
# LOG BAR-PER-BAR POSIZIONI APERTE (21/07/2026, parte 1)
# =====================================================================

def _last_closed_indicators(name: str, strategy: str, hist_cache: dict, now: datetime):
    """Ritorna la barra chiusa piu' recente con adx/ema/rsi gia'
    calcolati, riusando le stesse funzioni di generate_signals()/
    generate_mean_reversion_signals() gia' usate per i segnali —
    nessuna logica duplicata, solo ricalcolo (economico) sullo stesso
    hist_cache gia' scaricato una volta per ciclo."""
    hist = hist_cache[name]
    inst = eng.INSTRUMENTS[name]
    if strategy == "v6":
        signals = eng.generate_signals(hist, inst)
    else:
        signals = generate_mean_reversion_signals(hist, inst, mode=MR_MODE)
    closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
    if closed.empty:
        return None
    return closed.iloc[-1]


def log_position_bar(pos: dict, hist_cache: dict, current_price: float, now: datetime):
    """Aggiunge/aggiorna una riga in live_position_bars per la barra
    corrente di questa posizione. INSERT OR REPLACE su (position_id,
    bar_offset) — sicuro anche se il cron gira piu' volte sulla stessa
    barra da 30min (nessuna riga duplicata, l'ultima vince)."""
    entry_time = pd.Timestamp(pos["entry_time"]).to_pydatetime()
    bar_offset = max(0, int((now - entry_time).total_seconds() // (30 * 60)))

    row = _last_closed_indicators(pos["instrument"], pos["strategy"], hist_cache, now)
    if row is None:
        print(f"    [log-bar/{pos['instrument']}/{pos['strategy']}] nessuna barra chiusa disponibile, salto il log di questo ciclo.")
        return

    atr_at_entry = pos.get("atr_at_entry")
    inst = eng.INSTRUMENTS[pos["instrument"]]
    stop_distance = atr_at_entry * inst.atr_multiplier if atr_at_entry else None
    if stop_distance and stop_distance > 0:
        price_r = ((current_price - pos["entry_price"]) / stop_distance if pos["direction"] == "long"
                   else (pos["entry_price"] - current_price) / stop_distance)
    else:
        price_r = None

    ema_fast = float(row["ema_fast"]) if "ema_fast" in row.index and pd.notna(row["ema_fast"]) else None
    ema_slow = float(row["ema_slow"]) if "ema_slow" in row.index and pd.notna(row["ema_slow"]) else None
    rsi = float(row["rsi"]) if "rsi" in row.index and pd.notna(row["rsi"]) else None
    adx = float(row["adx"]) if pd.notna(row["adx"]) else None

    def fv(v):
        return "NULL" if v is None else str(v)

    d1_query(
        "INSERT OR REPLACE INTO live_position_bars "
        "(position_id, instrument, strategy, bar_offset, timestamp, open, high, low, close, "
        "current_price, price_r, adx, ema_fast, ema_slow, rsi, updated_at) VALUES ("
        f"{pos['id']}, '{pos['instrument']}', '{pos['strategy']}', {bar_offset}, "
        f"'{row['timestamp'].isoformat()}', {fv(float(row['open']) if pd.notna(row['open']) else None)}, "
        f"{fv(float(row['high']) if pd.notna(row['high']) else None)}, "
        f"{fv(float(row['low']) if pd.notna(row['low']) else None)}, "
        f"{fv(float(row['close']) if pd.notna(row['close']) else None)}, "
        f"{fv(current_price)}, {fv(price_r)}, {fv(adx)}, {fv(ema_fast)}, {fv(ema_slow)}, {fv(rsi)}, "
        f"'{now.isoformat()}')"
    )
    price_r_str = f"{price_r:+.2f}" if price_r is not None else "n/d"
    print(f"    [log-bar/{pos['instrument']}/{pos['strategy']}] bar_offset={bar_offset} price_r={price_r_str}")


def archive_position_bars(position_id: int, exit_reason: str, final_pnl: float, final_r_multiple):
    """Alla chiusura: copia tutte le righe live_position_bars di questa
    posizione in live_closed_position_bars (con esito finale allegato),
    poi CANCELLA le righe vive — evita bloat del D1 free tier, come
    deciso in chat 21/07/2026 (log vivo solo mentre la posizione e'
    aperta, archivio permanente separato dopo)."""
    rows = d1_query(f"SELECT * FROM live_position_bars WHERE position_id = {position_id}")
    if not rows:
        print(f"    [archive-bars] nessuna riga di log trovata per position_id={position_id} (posizione chiusa troppo in fretta per un ciclo di log, o mai loggata).")
        return

    now_iso = datetime.now(timezone.utc).isoformat()

    def fv(v):
        return "NULL" if v is None else (f"'{v}'" if isinstance(v, str) else str(v))

    for r in rows:
        d1_query(
            "INSERT OR REPLACE INTO live_closed_position_bars "
            "(position_id, instrument, strategy, bar_offset, timestamp, open, high, low, close, "
            "current_price, price_r, adx, ema_fast, ema_slow, rsi, exit_reason, final_pnl, "
            "final_r_multiple, archived_at) VALUES ("
            f"{r['position_id']}, '{r['instrument']}', '{r['strategy']}', {r['bar_offset']}, "
            f"'{r['timestamp']}', {fv(r['open'])}, {fv(r['high'])}, {fv(r['low'])}, {fv(r['close'])}, "
            f"{fv(r['current_price'])}, {fv(r['price_r'])}, {fv(r['adx'])}, {fv(r['ema_fast'])}, "
            f"{fv(r['ema_slow'])}, {fv(r['rsi'])}, '{exit_reason}', {fv(final_pnl)}, "
            f"{fv(final_r_multiple)}, '{now_iso}')"
        )
    d1_query(f"DELETE FROM live_position_bars WHERE position_id = {position_id}")
    print(f"    [archive-bars] {len(rows)} barre archiviate in live_closed_position_bars per position_id={position_id}, righe vive cancellate.")


# =====================================================================
# LIVELLO 3 — MARGINE REALE AGGREGATO (nuovo 21/07/2026, parte 3)
# =====================================================================

def compute_margin_state(session: IGSession) -> float | None:
    """Legge l'equity reale IG e somma il margine impegnato da TUTTE le
    posizioni aperte (V6+MR insieme, condividono lo stesso conto/margine
    reale anche se i pool di capitale sono contabili separati). Ritorna
    margine_libero, o None se l'equity non e' leggibile (fail-safe: il
    chiamante non blocca in questo caso, stesso pattern di apply_equity_cap)."""
    try:
        bal = session.get_account_balance()
        equity_reale = bal["equity"]
    except Exception as e:
        print(f"  [margine L3] impossibile leggere l'equity reale da IG ({e}) — "
              f"controllo margine disattivato questo ciclo (fail-safe, nessun blocco).")
        return None

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    instruments_needed = {p["instrument"] for p in open_positions}
    price_cache = {}
    for name in instruments_needed:
        try:
            price_cache[name] = session.get_price(name)
        except Exception as e:
            print(f"  [margine L3] impossibile leggere prezzo {name} ({e}) — "
                  f"posizioni su questo strumento escluse dal calcolo margine questo ciclo.")

    margine_impegnato = 0.0
    for pos in open_positions:
        price = price_cache.get(pos["instrument"])
        if price is None:
            continue
        current_price = price["bid"] if pos["direction"] == "long" else price["offer"]
        if current_price is None:
            continue
        inst = eng.INSTRUMENTS[pos["instrument"]]
        margine_impegnato += pos["size"] * current_price * inst.point_value * inst.margin_pct

    margine_libero = equity_reale - margine_impegnato
    print(f"  [margine L3] equity={equity_reale:.2f}  margine_impegnato(V6+MR)={margine_impegnato:.2f}  "
          f"margine_libero={margine_libero:.2f}")
    return margine_libero


# =====================================================================
# CAMPIONAMENTO SPREAD (spostato qui 21/07/2026) — riusa la sessione IG
# gia' aperta da questo ciclo di live_execute.py, invece di un secondo
# login separato (verify_ig_spread.py come workflow indipendente
# causava 401 Unauthorized intermittenti quando i due cron finivano
# per sovrapporsi/ravvicinarsi nel tempo — IG limita le sessioni
# concorrenti per la stessa API key sul conto demo, vedi chat
# 21/07/2026). verify_ig_spread.py/.yml restano nel repo solo per
# campionamenti manuali occasionali — NON vanno piu' agganciati a un
# cron separato.
# =====================================================================

_ASSUMED_SPREAD = {"DAX": 1.2, "FTSE100": 1.0}  # da engine.py, spread_fixed


def sample_spread(session: IGSession):
    """Campiona spread reale IG per DAX/FTSE100, stessa logica di
    verify_ig_spread.py ma riusando la sessione gia' loggata — nessun
    secondo login IG in questo ciclo."""
    now_iso = datetime.now(timezone.utc).isoformat()
    for instrument in ("DAX", "FTSE100"):
        try:
            price = session.get_price(instrument)
        except Exception as e:
            print(f"  [spread] impossibile leggere il prezzo {instrument} ({e}), salto.")
            continue
        bid, offer = price["bid"], price["offer"]
        if bid is None or offer is None:
            print(f"  [spread] {instrument} bid/offer non disponibili — mercato probabilmente chiuso.")
            continue
        spread = offer - bid
        assumed = _ASSUMED_SPREAD[instrument]
        d1_query(
            "INSERT INTO spread_samples (instrument, sample_time, bid, offer, spread, market_status) "
            f"VALUES ('{instrument}', '{now_iso}', {bid}, {offer}, {spread}, '{price['market_status']}')"
        )
        print(f"  [spread] {instrument} bid={bid} offer={offer} spread_reale={spread:.2f}pt "
              f"assunto={assumed}pt scarto={spread - assumed:+.2f}pt stato={price['market_status']}")


# =====================================================================
# TRACKING CANDIDATI LIVE (21/07/2026, parte 2)
# Registra OGNI segnale valido (eseguito o no) in research_v6/mr_
# candidates + live_v6/mr_candidates_tracking, con candidate_key nello
# STESSO formato usato da extract_v6/mr_candidates.py:
# f"{instrument}_{entry_time.isoformat()}" (entry_time = barra
# successiva alla barra di segnale) — cosi' un futuro merge/dedup con
# i dataset batch storici e' diretto, nessuna incoerenza di chiave.
# =====================================================================

_vix_cache: dict = {}


def get_current_vix_vix3m():
    """VIX/VIX3M piu' recenti disponibili (best-effort, non bloccante).
    Cache per singola esecuzione di main() — un solo fetch anche se
    richiesto da piu' candidati nello stesso ciclo."""
    if "vix" in _vix_cache:
        return _vix_cache["vix"], _vix_cache["vix3m"]
    try:
        vix = yf.download("^VIX", period="5d", progress=False)
        vix3m = yf.download("^VIX3M", period="5d", progress=False)
        for df in (vix, vix3m):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        vix_val = float(vix["Close"].iloc[-1]) if not vix.empty else None
        vix3m_val = float(vix3m["Close"].iloc[-1]) if not vix3m.empty else None
    except Exception as e:
        print(f"  [vix] impossibile scaricare VIX/VIX3M ({e}) — campi lasciati NULL.")
        vix_val, vix3m_val = None, None
    _vix_cache["vix"], _vix_cache["vix3m"] = vix_val, vix3m_val
    return vix_val, vix3m_val


def count_consecutive_backward(bool_series: pd.Series, end_idx: int) -> int:
    """Identica a quella in extract_v6/mr_candidates.py — nessuna
    reimplementazione divergente."""
    i = end_idx
    count = 0
    while i >= 0 and bool(bool_series.iloc[i]):
        count += 1
        i -= 1
    return count


def _fv(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


def record_candidate_v6(name: str, signals: pd.DataFrame, last_closed_idx: int,
                         now: datetime, was_executed: int, skip_reason: str | None,
                         matched_trade_key: str | None = None) -> str:
    prev_bar = signals.iloc[last_closed_idx]
    direction = prev_bar["signal"]
    entry_time_dt = prev_bar["timestamp"] + timedelta(minutes=30)
    candidate_key = f"{name}_{entry_time_dt.isoformat()}"

    breakout_level = prev_bar["rolling_high"] if direction == "long" else prev_bar["rolling_low"]
    breakout_dist = (((prev_bar["close"] - breakout_level) if direction == "long"
                       else (breakout_level - prev_bar["close"]))
                      if pd.notna(breakout_level) else None)

    persistence = count_consecutive_backward(signals["signal"] == direction, last_closed_idx)
    adx_regime_age = count_consecutive_backward(
        signals["adx"] > eng.PARAMS.adx_min_context, last_closed_idx)

    vix_val, vix3m_val = get_current_vix_vix3m()

    cols = ("candidate_key, instrument, direction, entry_time, signal_bar_time, "
            "adx_at_entry, atr_at_entry, ema_fast_entry, ema_slow_entry, "
            "ema_broad_fast_entry, ema_broad_slow_entry, close_entry, "
            "breakout_level_entry, breakout_distance_pts, persistence_bars, "
            "adx_regime_age_bars, vix_entry, vix3m_entry, was_executed, "
            "matched_trade_key, skip_reason")

    vals = (
        f"{_fv(candidate_key)}, {_fv(name)}, {_fv(direction)}, {_fv(entry_time_dt.isoformat())}, "
        f"{_fv(prev_bar['timestamp'].isoformat())}, "
        f"{_fv(float(prev_bar['adx']) if pd.notna(prev_bar['adx']) else None)}, "
        f"{_fv(float(prev_bar['atr']) if pd.notna(prev_bar['atr']) else None)}, "
        f"{_fv(float(prev_bar['ema_fast']) if pd.notna(prev_bar['ema_fast']) else None)}, "
        f"{_fv(float(prev_bar['ema_slow']) if pd.notna(prev_bar['ema_slow']) else None)}, "
        f"{_fv(float(prev_bar['ema_broad_fast']) if pd.notna(prev_bar['ema_broad_fast']) else None)}, "
        f"{_fv(float(prev_bar['ema_broad_slow']) if pd.notna(prev_bar['ema_broad_slow']) else None)}, "
        f"{_fv(float(prev_bar['close']))}, "
        f"{_fv(float(breakout_level) if pd.notna(breakout_level) else None)}, "
        f"{_fv(float(breakout_dist) if breakout_dist is not None and pd.notna(breakout_dist) else None)}, "
        f"{persistence}, {adx_regime_age}, {_fv(vix_val)}, {_fv(vix3m_val)}, {was_executed}, "
        f"{_fv(matched_trade_key)}, {_fv(skip_reason)}"
    )

    d1_query(f"INSERT OR REPLACE INTO research_v6_candidates ({cols}, extraction_run_at) "
             f"VALUES ({vals}, {_fv(now.isoformat())})")
    d1_query(f"INSERT OR REPLACE INTO live_v6_candidates_tracking ({cols}, created_at) "
             f"VALUES ({vals}, {_fv(now.isoformat())})")
    print(f"    [candidate/V6/{name}] {candidate_key} "
          f"(was_executed={was_executed}, skip_reason={skip_reason})")
    return candidate_key


def record_candidate_mr(name: str, signals: pd.DataFrame, last_closed_idx: int,
                         now: datetime, was_executed: int, skip_reason: str | None,
                         matched_trade_key: str | None = None) -> str:
    prev_bar = signals.iloc[last_closed_idx]
    direction = prev_bar["signal"]
    entry_time_dt = prev_bar["timestamp"] + timedelta(minutes=30)
    candidate_key = f"{name}_{entry_time_dt.isoformat()}"

    persistence = count_consecutive_backward(signals["signal"] == direction, last_closed_idx)
    # NB: stessa soglia di extract_mr_candidates.py (eng.PARAMS.adx_min_context,
    # non ADX_THRESHOLD di mean_reversion_signals — stesso valore numerico,
    # costante diversa, mantenuta per coerenza esatta col dataset batch)
    adx_regime_age = count_consecutive_backward(
        signals["adx"] < eng.PARAMS.adx_min_context, last_closed_idx)

    vix_val, vix3m_val = get_current_vix_vix3m()
    rsi_val = float(prev_bar["rsi"]) if "rsi" in prev_bar.index and pd.notna(prev_bar["rsi"]) else None

    cols = ("candidate_key, instrument, direction, entry_time, signal_bar_time, "
            "adx_at_entry, atr_at_entry, rsi_at_entry, close_entry, "
            "persistence_bars, adx_regime_age_bars, vix_entry, vix3m_entry, "
            "was_executed, matched_trade_key, skip_reason")

    vals = (
        f"{_fv(candidate_key)}, {_fv(name)}, {_fv(direction)}, {_fv(entry_time_dt.isoformat())}, "
        f"{_fv(prev_bar['timestamp'].isoformat())}, "
        f"{_fv(float(prev_bar['adx']) if pd.notna(prev_bar['adx']) else None)}, "
        f"{_fv(float(prev_bar['atr']) if pd.notna(prev_bar['atr']) else None)}, "
        f"{_fv(rsi_val)}, {_fv(float(prev_bar['close']))}, "
        f"{persistence}, {adx_regime_age}, {_fv(vix_val)}, {_fv(vix3m_val)}, {was_executed}, "
        f"{_fv(matched_trade_key)}, {_fv(skip_reason)}"
    )

    d1_query(f"INSERT OR REPLACE INTO research_mr_candidates ({cols}, extraction_run_at) "
             f"VALUES ({vals}, {_fv(now.isoformat())})")
    d1_query(f"INSERT OR REPLACE INTO live_mr_candidates_tracking ({cols}, created_at) "
             f"VALUES ({vals}, {_fv(now.isoformat())})")
    print(f"    [candidate/MR/{name}] {candidate_key} "
          f"(was_executed={was_executed}, skip_reason={skip_reason})")
    return candidate_key


def log_candidate_bars(strategy: str, hist_cache: dict, now: datetime):
    """Ad ogni ciclo, scrive la prossima barra disponibile per ogni
    candidato ancora 'in tracking' (bar_offset non ancora a 48). Nessun
    flag is_complete separato: un candidato e' completo quando esiste
    la riga con bar_offset=48 nella tabella *_bars — stesso principio
    gia' usato per live_position_bars -> live_closed_position_bars."""
    tracking_table = "live_v6_candidates_tracking" if strategy == "v6" else "live_mr_candidates_tracking"
    bars_table = "live_v6_candidate_bars" if strategy == "v6" else "live_mr_candidate_bars"

    rows = d1_query(f"SELECT candidate_key, instrument, entry_time FROM {tracking_table}")
    if not rows:
        return

    for r in rows:
        max_offset_rows = d1_query(
            f"SELECT MAX(bar_offset) as mx FROM {bars_table} WHERE candidate_key = '{r['candidate_key']}'"
        )
        current_max = max_offset_rows[0]["mx"] if max_offset_rows else None
        next_offset = 0 if current_max is None else int(current_max) + 1
        if next_offset > 48:
            continue  # gia' completo, nulla da fare

        entry_time_dt = pd.Timestamp(r["entry_time"]).to_pydatetime()
        target_ts = entry_time_dt + timedelta(minutes=30 * next_offset)
        if target_ts > now:
            continue  # barra target non ancora chiusa

        name = r["instrument"]
        hist = hist_cache.get(name)
        if hist is None:
            continue
        inst = eng.INSTRUMENTS[name]
        if strategy == "v6":
            signals = eng.generate_signals(hist, inst)
        else:
            signals = generate_mean_reversion_signals(hist, inst, mode=MR_MODE)

        match = signals[signals["timestamp"] == pd.Timestamp(target_ts, tz="UTC")]
        if match.empty:
            continue
        bar = match.iloc[0]

        cols = "candidate_key, bar_offset, timestamp, open, high, low, close, price_r, adx"
        if strategy == "v6":
            cols += ", ema_fast, ema_slow"
        else:
            cols += ", rsi, ema_fast, ema_slow"

        vals = (
            f"{_fv(r['candidate_key'])}, {next_offset}, {_fv(bar['timestamp'].isoformat())}, "
            f"{_fv(float(bar['open']) if pd.notna(bar['open']) else None)}, "
            f"{_fv(float(bar['high']) if pd.notna(bar['high']) else None)}, "
            f"{_fv(float(bar['low']) if pd.notna(bar['low']) else None)}, "
            f"{_fv(float(bar['close']) if pd.notna(bar['close']) else None)}, "
            f"NULL, "  # price_r: da calcolare con lo stop_distance del candidato se serve in futuro
            f"{_fv(float(bar['adx']) if pd.notna(bar['adx']) else None)}"
        )
        if strategy == "v6":
            vals += (f", {_fv(float(bar['ema_fast']) if pd.notna(bar['ema_fast']) else None)}, "
                      f"{_fv(float(bar['ema_slow']) if pd.notna(bar['ema_slow']) else None)}")
        else:
            rsi_v = float(bar["rsi"]) if "rsi" in bar.index and pd.notna(bar["rsi"]) else None
            vals += (f", {_fv(rsi_v)}, "
                      f"{_fv(float(bar['ema_fast']) if pd.notna(bar['ema_fast']) else None)}, "
                      f"{_fv(float(bar['ema_slow']) if pd.notna(bar['ema_slow']) else None)}")

        d1_query(f"INSERT OR REPLACE INTO {bars_table} ({cols}) VALUES ({vals})")
    print(f"  [candidate-bars/{strategy}] {len(rows)} candidati in tracking, barre aggiornate dove disponibili.")


def apply_monthly_consolidation_if_needed(today_str: str, prev_state: dict) -> dict:
    """Accantonamento PER POOL INDIPENDENTE (corretto 22/07/2026 — bug
    scoperto in chat: la versione precedente consolidava su capitale
    COMBINATO v6+mr, e siccome V6 genera quasi tutto il guadagno mentre
    MR resta piatto, ogni consolidamento erodeva PROPORZIONALMENTE
    anche la quota MR anche se MR non aveva contribuito nulla al
    superamento soglia. Verificato con simulazione continua 11 anni:
    pool MR 600EUR -> 101EUR (quasi azzerato) con la vecchia logica,
    contro 600EUR -> 541EUR (sostanzialmente stabile) con questa
    versione per-pool. Coerente col principio guida gia' stabilito nel
    progetto: split capital preferito su shared/router, pool separati
    proteggono meglio il drawdown della massimizzazione del pool
    combinato. Ogni pool ha ora reference/threshold PROPRI — nessun
    pool tocca mai l'altro."""
    capital_v6 = prev_state.get("capital_current_v6")
    capital_mr = prev_state.get("capital_current_mr")
    if capital_v6 is None or capital_mr is None:
        legacy_capital = prev_state["capital_current"]
        capital_v6 = legacy_capital * SPLIT_V6_PCT
        capital_mr = legacy_capital * SPLIT_MR_PCT

    accantonato_v6 = prev_state.get("accantonato_v6", 0.0) or 0.0
    accantonato_mr = prev_state.get("accantonato_mr", 0.0) or 0.0
    valvola_budget_v6 = prev_state.get("valvola_budget_v6", 0.0) or 0.0
    valvola_consumato_v6 = prev_state.get("valvola_consumato_v6", 0.0) or 0.0
    valvola_rif_v6 = prev_state.get("valvola_accantonato_riferimento_v6", 0.0) or 0.0
    valvola_budget_mr = prev_state.get("valvola_budget_mr", 0.0) or 0.0
    valvola_consumato_mr = prev_state.get("valvola_consumato_mr", 0.0) or 0.0
    valvola_rif_mr = prev_state.get("valvola_accantonato_riferimento_mr", 0.0) or 0.0

    if not prev_state.get("accantonamento_attivo", 1):
        return {
            "capital_v6": capital_v6, "capital_mr": capital_mr,
            "accantonato_v6": accantonato_v6, "accantonato_mr": accantonato_mr,
            "reference_v6": prev_state.get("consolidamento_reference_v6") or capital_v6,
            "threshold_v6": prev_state.get("consolidamento_threshold_v6") or capital_v6 * THRESHOLD_MULT,
            "reference_mr": prev_state.get("consolidamento_reference_mr") or capital_mr,
            "threshold_mr": prev_state.get("consolidamento_threshold_mr") or capital_mr * THRESHOLD_MULT,
            "valvola_budget_v6": valvola_budget_v6, "valvola_consumato_v6": valvola_consumato_v6,
            "valvola_accantonato_riferimento_v6": valvola_rif_v6,
            "valvola_budget_mr": valvola_budget_mr, "valvola_consumato_mr": valvola_consumato_mr,
            "valvola_accantonato_riferimento_mr": valvola_rif_mr,
        }

    reference_v6 = prev_state.get("consolidamento_reference_v6") or capital_v6
    threshold_v6 = prev_state.get("consolidamento_threshold_v6") or (reference_v6 * THRESHOLD_MULT)
    reference_mr = prev_state.get("consolidamento_reference_mr") or capital_mr
    threshold_mr = prev_state.get("consolidamento_threshold_mr") or (reference_mr * THRESHOLD_MULT)

    prev_month = prev_state["trade_date"][:7]
    this_month = today_str[:7]

    if this_month != prev_month:
        # --- V6, indipendente ---
        while capital_v6 > threshold_v6:
            gain = capital_v6 - reference_v6
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            capital_v6 -= consolidated
            accantonato_v6 += consolidated
            reference_v6 = capital_v6
            threshold_v6 = reference_v6 * THRESHOLD_MULT
            print(f"[accantonamento/V6] Consolidati {consolidated:.2f} EUR al cambio mese "
                  f"({prev_month} -> {this_month}). Investito V6: {capital_v6:.2f}  "
                  f"Accantonato V6: {accantonato_v6:.2f}")

        # --- MR, indipendente ---
        while capital_mr > threshold_mr:
            gain = capital_mr - reference_mr
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            capital_mr -= consolidated
            accantonato_mr += consolidated
            reference_mr = capital_mr
            threshold_mr = reference_mr * THRESHOLD_MULT
            print(f"[accantonamento/MR] Consolidati {consolidated:.2f} EUR al cambio mese "
                  f"({prev_month} -> {this_month}). Investito MR: {capital_mr:.2f}  "
                  f"Accantonato MR: {accantonato_mr:.2f}")

        # --- valvola V6, budget dal SOLO accantonato V6 ---
        if accantonato_v6 > valvola_rif_v6:
            vecchio = valvola_rif_v6
            valvola_budget_v6 = VALVOLA_PCT * accantonato_v6
            valvola_consumato_v6 = 0.0
            valvola_rif_v6 = accantonato_v6
            print(f"[valvola/V6] Reset al cambio mese: accantonato V6 {accantonato_v6:.2f} > "
                  f"riferimento precedente {vecchio:.2f} -> nuovo budget {valvola_budget_v6:.2f} EUR")

        # --- valvola MR, budget dal SOLO accantonato MR ---
        if accantonato_mr > valvola_rif_mr:
            vecchio = valvola_rif_mr
            valvola_budget_mr = VALVOLA_PCT * accantonato_mr
            valvola_consumato_mr = 0.0
            valvola_rif_mr = accantonato_mr
            print(f"[valvola/MR] Reset al cambio mese: accantonato MR {accantonato_mr:.2f} > "
                  f"riferimento precedente {vecchio:.2f} -> nuovo budget {valvola_budget_mr:.2f} EUR")

    return {
        "capital_v6": capital_v6, "capital_mr": capital_mr,
        "accantonato_v6": accantonato_v6, "accantonato_mr": accantonato_mr,
        "reference_v6": reference_v6, "threshold_v6": threshold_v6,
        "reference_mr": reference_mr, "threshold_mr": threshold_mr,
        "valvola_budget_v6": valvola_budget_v6, "valvola_consumato_v6": valvola_consumato_v6,
        "valvola_accantonato_riferimento_v6": valvola_rif_v6,
        "valvola_budget_mr": valvola_budget_mr, "valvola_consumato_mr": valvola_consumato_mr,
        "valvola_accantonato_riferimento_mr": valvola_rif_mr,
    }


CONSOLIDATE_PCT = 0.4      # opzione 3 mensile, validata su 5 periodi ufficiali il 16-17/07/2026
THRESHOLD_MULT = 1.5


def get_or_create_today_state(today_str: str) -> dict:
    rows = d1_query(f"SELECT * FROM live_daily_state WHERE trade_date = '{today_str}'")
    if rows:
        return rows[0]

    prev_rows = d1_query("SELECT * FROM live_daily_state ORDER BY trade_date DESC LIMIT 1")

    if prev_rows:
        updated = apply_monthly_consolidation_if_needed(today_str, prev_rows[0])
        start_v6, start_mr = updated["capital_v6"], updated["capital_mr"]
        acc_v6, acc_mr = updated["accantonato_v6"], updated["accantonato_mr"]
        reference_v6, threshold_v6 = updated["reference_v6"], updated["threshold_v6"]
        reference_mr, threshold_mr = updated["reference_mr"], updated["threshold_mr"]
        vb_v6, vc_v6, vr_v6 = updated["valvola_budget_v6"], updated["valvola_consumato_v6"], updated["valvola_accantonato_riferimento_v6"]
        vb_mr, vc_mr, vr_mr = updated["valvola_budget_mr"], updated["valvola_consumato_mr"], updated["valvola_accantonato_riferimento_mr"]
    else:
        start_v6 = CAPITAL0_DEFAULT * SPLIT_V6_PCT
        start_mr = CAPITAL0_DEFAULT * SPLIT_MR_PCT
        acc_v6, acc_mr = 0.0, 0.0
        reference_v6, threshold_v6 = start_v6, start_v6 * THRESHOLD_MULT
        reference_mr, threshold_mr = start_mr, start_mr * THRESHOLD_MULT
        vb_v6 = vc_v6 = vr_v6 = 0.0
        vb_mr = vc_mr = vr_mr = 0.0

    combined = start_v6 + start_mr
    accantonato_totale = acc_v6 + acc_mr
    d1_query(
        "INSERT INTO live_daily_state "
        "(trade_date, account_type, capital_start_of_day, capital_current, "
        "capital_start_of_day_v6, capital_current_v6, capital_start_of_day_mr, capital_current_mr, "
        "accantonato, accantonato_v6, accantonato_mr, "
        "consolidamento_reference_v6, consolidamento_threshold_v6, "
        "consolidamento_reference_mr, consolidamento_threshold_mr, accantonamento_attivo, "
        "valvola_budget_v6, valvola_consumato_v6, valvola_accantonato_riferimento_v6, "
        "valvola_budget_mr, valvola_consumato_mr, valvola_accantonato_riferimento_mr) "
        f"VALUES ('{today_str}', 'demo', {combined}, {combined}, "
        f"{start_v6}, {start_v6}, {start_mr}, {start_mr}, "
        f"{accantonato_totale}, {acc_v6}, {acc_mr}, "
        f"{reference_v6}, {threshold_v6}, {reference_mr}, {threshold_mr}, 1, "
        f"{vb_v6}, {vc_v6}, {vr_v6}, {vb_mr}, {vc_mr}, {vr_mr})"
    )
    print(f"Creato nuovo record live_daily_state per {today_str}: "
          f"V6={start_v6:.2f} EUR (accantonato V6={acc_v6:.2f}), "
          f"MR={start_mr:.2f} EUR (accantonato MR={acc_mr:.2f})")
    return {
        "trade_date": today_str, "account_type": "demo",
        "capital_current_v6": start_v6, "capital_current_mr": start_mr,
        "capital_start_of_day_v6": start_v6, "capital_start_of_day_mr": start_mr,
        "accantonato": accantonato_totale, "accantonato_v6": acc_v6, "accantonato_mr": acc_mr,
        "orders_today_v6": 0, "orders_today_mr": 0,
        "kill_switch_triggered_v6": 0, "kill_switch_triggered_mr": 0,
        "kill_switch_threshold_pct": -4.0,
        "valvola_budget_v6": vb_v6, "valvola_consumato_v6": vc_v6, "valvola_accantonato_riferimento_v6": vr_v6,
        "valvola_budget_mr": vb_mr, "valvola_consumato_mr": vc_mr, "valvola_accantonato_riferimento_mr": vr_mr,
    }


def get_today_state(today_str: str) -> dict:
    return get_or_create_today_state(today_str)


def apply_equity_cap(session: IGSession, today_str: str) -> dict:
    """LIVELLO 1. INVARIATO dal 19/07/2026."""
    day_state = get_today_state(today_str)
    try:
        bal = session.get_account_balance()
        equity_reale = bal["equity"]
    except Exception as e:
        print(f"[tetto L1] impossibile leggere l'equity reale da IG ({e}) — "
              f"salto il controllo questo ciclo, uso i valori D1 esistenti senza modifiche.")
        return day_state

    accantonato = day_state.get("accantonato", 0.0) or 0.0
    capitale_investibile_totale = max(0.0, equity_reale - accantonato)

    capital_v6 = day_state.get("capital_current_v6") or 0.0
    capital_mr = day_state.get("capital_current_mr") or 0.0
    combined = capital_v6 + capital_mr

    if combined > capitale_investibile_totale + 0.01:
        fraction = capitale_investibile_totale / combined if combined > 0 else 0.0
        new_v6 = capital_v6 * fraction
        new_mr = capital_mr * fraction
        print(f"[tetto L1] *** SCALING *** capitale tracciato ({combined:.2f}) supera "
              f"equity-accantonato ({capitale_investibile_totale:.2f}) — "
              f"riduco V6 {capital_v6:.2f}->{new_v6:.2f}, MR {capital_mr:.2f}->{new_mr:.2f}")
        d1_query(
            f"UPDATE live_daily_state SET capital_current_v6 = {new_v6}, "
            f"capital_current_mr = {new_mr}, equity_reale_ultima = {equity_reale}, "
            f"capitale_investibile_totale = {capitale_investibile_totale} "
            f"WHERE trade_date = '{today_str}'"
        )
        day_state["capital_current_v6"] = new_v6
        day_state["capital_current_mr"] = new_mr
    else:
        d1_query(
            f"UPDATE live_daily_state SET equity_reale_ultima = {equity_reale}, "
            f"capitale_investibile_totale = {capitale_investibile_totale} "
            f"WHERE trade_date = '{today_str}'"
        )
        print(f"  [tetto L1] equity={equity_reale:.2f}  accantonato={accantonato:.2f}  "
              f"capitale_investibile_totale={capitale_investibile_totale:.2f}  "
              f"(capitale tracciato={combined:.2f}, entro il tetto)")

    return day_state


def try_valvola(today_str: str, day_state: dict, risk_amount_needed_for_min: float,
                 pool_capital: float, risk_pct: float, pool: str) -> tuple[float, bool]:
    """LIVELLO 2. Corretto 22/07/2026: budget/consumato ora PER POOL
    (colonna _v6 o _mr in base al parametro `pool`), coerente con la
    correzione dell'accantonamento — la valvola V6 attinge solo
    dall'accantonato V6, quella MR solo dall'accantonato MR. Prima
    attingevano da un budget condiviso, stesso bug concettuale del
    consolidamento (un pool poteva consumare budget generato
    dall'altro)."""
    extra_capital_needed = risk_amount_needed_for_min / risk_pct - pool_capital
    if extra_capital_needed <= 0:
        return pool_capital, False

    budget_field = f"valvola_budget_{pool}"
    consumato_field = f"valvola_consumato_{pool}"

    budget = day_state.get(budget_field, 0.0) or 0.0
    consumato = day_state.get(consumato_field, 0.0) or 0.0
    budget_residuo = max(0.0, budget - consumato)

    draw = min(extra_capital_needed, budget_residuo)
    if draw > 0:
        nuovo_consumato = consumato + draw
        d1_query(
            f"UPDATE live_daily_state SET {consumato_field} = {nuovo_consumato} "
            f"WHERE trade_date = '{today_str}'"
        )
        day_state[consumato_field] = nuovo_consumato
        print(f"    [valvola/{pool}] prelevati {draw:.2f} EUR (budget residuo ora "
              f"{budget_residuo - draw:.2f}/{budget:.2f} EUR)")

    nuovo_capitale = pool_capital + draw
    ancora_insufficiente = (nuovo_capitale * risk_pct) < risk_amount_needed_for_min
    return nuovo_capitale, ancora_insufficiente


def manage_open_positions(session: IGSession, today_str: str, hist_cache: dict):
    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open'")
    if not open_positions:
        print("Nessuna posizione aperta da gestire.")
        return

    day_state = get_today_state(today_str)
    capital_v6 = day_state["capital_current_v6"]
    capital_mr = day_state["capital_current_mr"]
    now = datetime.now(timezone.utc)

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

        # --- LOG BAR-PER-BAR (nuovo 21/07/2026): sempre, indipendentemente
        # da un'eventuale chiusura piu' sotto in questo stesso ciclo ---
        log_position_bar(pos, hist_cache, current_price, now)

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

        # --- ARCHIVIAZIONE LOG BAR-PER-BAR (nuovo 21/07/2026) ---
        r_multiple = pnl / pos["risk_amount"] if pos.get("risk_amount") else None
        archive_position_bars(pos["id"], exit_reason, pnl, r_multiple)

        if pos["strategy"] == "mean_reversion":
            capital_mr += pnl
            d1_query(f"UPDATE live_daily_state SET capital_current_mr = {capital_mr} WHERE trade_date = '{today_str}'")
        else:
            capital_v6 += pnl
            d1_query(f"UPDATE live_daily_state SET capital_current_v6 = {capital_v6} WHERE trade_date = '{today_str}'")

        print(f"  [{pos['instrument']}/{pos['strategy']}] chiusa: pnl={pnl:+.2f} EUR, "
              f"nuovo capitale pool {pos['strategy']}={(capital_mr if pos['strategy']=='mean_reversion' else capital_v6):.2f} EUR")


def check_and_apply_kill_switches(session: IGSession, today_str: str):
    """Kill switch giornaliero, SEPARATO per pool. INVARIATO dal 18/07/2026."""
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


def detect_and_open_signals_v6(session: IGSession, today_str: str, hist_cache: dict,
                                margine_libero: float | None) -> float | None:
    day_state = get_today_state(today_str)
    now = datetime.now(timezone.utc)
    capital = day_state["capital_current_v6"]

    kill_switch_on = bool(day_state.get("kill_switch_triggered_v6"))
    orders_limit_hit = (day_state.get("orders_today_v6") or 0) >= eng.PARAMS.max_new_orders_per_day

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open' AND strategy = 'v6'")
    slots_full = len(open_positions) >= eng.PARAMS.max_concurrent_positions
    open_instruments = {p["instrument"] for p in open_positions}

    if kill_switch_on:
        print("[V6] Kill switch attivo oggi — nessun nuovo ordine (segnali comunque calcolati e tracciati).")
    if orders_limit_hit:
        print("[V6] Limite ordini/giorno raggiunto — nessun nuovo ordine (segnali comunque calcolati e tracciati).")
    if slots_full:
        print("[V6] Nessuno slot concorrente libero — nessun nuovo ordine (segnali comunque calcolati e tracciati).")

    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        hist = hist_cache[name]
        signals = eng.generate_signals(hist, inst)
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            continue
        last_closed_idx = closed.index[-1]
        last_closed = signals.loc[last_closed_idx]

        # --- DIAGNOSTICA (nuovo 20/07/2026): sempre stampata, segnale o no ---
        print(_diag_v6(name, last_closed))

        sig = last_closed["signal"]
        if sig not in ("long", "short"):
            print(f"  [V6/{name}] nessun segnale.")
            continue

        # --- determina motivo di skip, se presente (priorita' fissa) ---
        skip_reason = None
        if kill_switch_on:
            skip_reason = "kill_switch_attivo"
        elif orders_limit_hit:
            skip_reason = "limite_ordini_giorno"
        elif slots_full:
            skip_reason = "slot_pieno"
        elif name in open_instruments:
            skip_reason = "strumento_gia_in_posizione"

        if skip_reason:
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0, skip_reason=skip_reason)
            continue

        atr = last_closed["atr"]
        if pd.isna(atr):
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="atr_non_disponibile")
            print(f"  [V6/{name}] ATR non disponibile, salto.")
            continue

        try:
            price = session.get_price(name)
        except Exception as e:
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="prezzo_non_disponibile")
            print(f"  [V6/{name}] impossibile leggere il prezzo per l'ordine ({e}), salto.")
            continue
        entry_price = price["offer"] if sig == "long" else price["bid"]
        if entry_price is None:
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="prezzo_non_disponibile")
            print(f"  [V6/{name}] prezzo non disponibile, salto.")
            continue

        stop_distance_pts = atr * inst.atr_multiplier
        limit_distance_pts = stop_distance_pts * eng.PARAMS.rr_target
        risk_amount = capital * inst.risk_pct
        size = risk_amount / (stop_distance_pts * inst.point_value)
        forced_min = False

        if size < inst.min_tradable_size:
            risk_amount_needed = inst.min_tradable_size * stop_distance_pts * inst.point_value
            capital, ancora_insufficiente = try_valvola(
                today_str, day_state, risk_amount_needed, capital, inst.risk_pct, pool="v6")
            risk_amount = capital * inst.risk_pct
            size = risk_amount / (stop_distance_pts * inst.point_value)
            if ancora_insufficiente or size < inst.min_tradable_size:
                size = inst.min_tradable_size
                forced_min = True

        # --- LIVELLO 3 (nuovo 21/07/2026): controllo margine reale aggregato
        # V6+MR, dopo il sizing finale (risk/min-forcing/valvola). Niente
        # "forza al minimo" qui: forzare la size aumenterebbe il margine
        # richiesto, l'opposto di quanto serve se il margine e' il vincolo. ---
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        if margine_libero is not None and margin_required > margine_libero:
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="margine_insufficiente")
            print(f"  [V6/{name}] margine insufficiente: richiesto {margin_required:.2f} EUR, "
                  f"libero {margine_libero:.2f} EUR — salto.")
            continue

        direction = "BUY" if sig == "long" else "SELL"
        print(f"  [V6/{name}] segnale {sig.upper()} — size={size:.2f} "
              f"(forzata al minimo={forced_min}) stop={stop_distance_pts:.1f}pt "
              f"target={limit_distance_pts:.1f}pt margine_richiesto={margin_required:.2f} EUR (dry_run={DRY_RUN})")

        result = session.place_order(
            instrument=name, direction=direction, size=size,
            stop_distance=stop_distance_pts, limit_distance=limit_distance_pts, dry_run=DRY_RUN,
        )

        if DRY_RUN:
            record_candidate_v6(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="dry_run")
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
            f"UPDATE live_daily_state SET orders_today_v6 = orders_today_v6 + 1, "
            f"capital_current_v6 = {capital} WHERE trade_date = '{today_str}'"
        )
        candidate_key = record_candidate_v6(name, signals, last_closed_idx, now, was_executed=1,
                                             skip_reason=None)
        print(f"    Posizione V6 aperta su IG, deal_id={deal_id}, candidate_key={candidate_key}")

        # --- Aggiorna margine_libero DENTRO lo stesso ciclo: un ordine
        # reale appena piazzato consuma margine anche per i controlli
        # successivi (prossimo strumento V6, poi tutto il ciclo MR) ---
        if margine_libero is not None:
            margine_libero -= margin_required

    return margine_libero


def detect_and_open_signals_mr(session: IGSession, today_str: str, hist_cache: dict,
                                margine_libero: float | None) -> float | None:
    """Identica a detect_and_open_signals_v6 nella struttura, con due
    differenze intenzionali: segnale generate_mean_reversion_signals()
    variante RSI, e size che SALTA (non forza) sotto il minimo se la
    valvola non basta a coprire il gap."""
    day_state = get_today_state(today_str)
    now = datetime.now(timezone.utc)
    capital = day_state["capital_current_mr"]

    kill_switch_on = bool(day_state.get("kill_switch_triggered_mr"))
    orders_limit_hit = (day_state.get("orders_today_mr") or 0) >= eng.PARAMS.max_new_orders_per_day

    open_positions = d1_query("SELECT * FROM live_positions WHERE status = 'open' AND strategy = 'mean_reversion'")
    slots_full = len(open_positions) >= eng.PARAMS.max_concurrent_positions
    open_instruments = {p["instrument"] for p in open_positions}

    if kill_switch_on:
        print("[MR] Kill switch attivo oggi — nessun nuovo ordine (segnali comunque calcolati e tracciati).")
    if orders_limit_hit:
        print("[MR] Limite ordini/giorno raggiunto — nessun nuovo ordine (segnali comunque calcolati e tracciati).")
    if slots_full:
        print("[MR] Nessuno slot concorrente libero — nessun nuovo ordine (segnali comunque calcolati e tracciati).")

    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        hist = hist_cache[name]
        signals = generate_mean_reversion_signals(hist, inst, mode=MR_MODE)
        closed = signals[signals["timestamp"] + timedelta(minutes=30) <= now]
        if closed.empty:
            continue
        last_closed_idx = closed.index[-1]
        last_closed = signals.loc[last_closed_idx]

        # --- DIAGNOSTICA (nuovo 20/07/2026): sempre stampata, segnale o no ---
        print(_diag_mr(name, last_closed, MR_MODE))

        sig = last_closed["signal"]
        if sig not in ("long", "short"):
            print(f"  [MR/{name}] nessun segnale.")
            continue

        skip_reason = None
        if kill_switch_on:
            skip_reason = "kill_switch_attivo"
        elif orders_limit_hit:
            skip_reason = "limite_ordini_giorno"
        elif slots_full:
            skip_reason = "slot_pieno"
        elif name in open_instruments:
            skip_reason = "strumento_gia_in_posizione"

        if skip_reason:
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0, skip_reason=skip_reason)
            continue

        atr = last_closed["atr"]
        if pd.isna(atr):
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="atr_non_disponibile")
            print(f"  [MR/{name}] ATR non disponibile, salto.")
            continue

        try:
            price = session.get_price(name)
        except Exception as e:
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="prezzo_non_disponibile")
            print(f"  [MR/{name}] impossibile leggere il prezzo per l'ordine ({e}), salto.")
            continue
        entry_price = price["offer"] if sig == "long" else price["bid"]
        if entry_price is None:
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="prezzo_non_disponibile")
            print(f"  [MR/{name}] prezzo non disponibile, salto.")
            continue

        stop_distance_pts = atr * inst.atr_multiplier
        limit_distance_pts = stop_distance_pts * eng.PARAMS.rr_target
        risk_amount = capital * inst.risk_pct
        size = risk_amount / (stop_distance_pts * inst.point_value)

        if size < inst.min_tradable_size:
            risk_amount_needed = inst.min_tradable_size * stop_distance_pts * inst.point_value
            capital, ancora_insufficiente = try_valvola(
                today_str, day_state, risk_amount_needed, capital, inst.risk_pct, pool="mr")
            risk_amount = capital * inst.risk_pct
            size = risk_amount / (stop_distance_pts * inst.point_value)

            if ancora_insufficiente or size < inst.min_tradable_size:
                record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                     skip_reason="size_sotto_minimo_dopo_valvola")
                print(f"  [MR/{name}] segnale {sig.upper()} SALTATO — size calcolata {size:.3f} "
                      f"sotto il minimo {inst.min_tradable_size} anche dopo la valvola "
                      f"(capitale pool MR={capital:.2f} EUR insufficiente). "
                      f"Comportamento intenzionale, vedi engine_mean_reversion.py.")
                continue

        # --- LIVELLO 3 (nuovo 21/07/2026): stesso controllo di V6, vedi
        # commento li' per il dettaglio. Qui non c'e' mai "forza al minimo"
        # da annullare (MR gia' salta sotto il minimo), solo lo skip. ---
        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        if margine_libero is not None and margin_required > margine_libero:
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="margine_insufficiente")
            print(f"  [MR/{name}] margine insufficiente: richiesto {margin_required:.2f} EUR, "
                  f"libero {margine_libero:.2f} EUR — salto.")
            continue

        direction = "BUY" if sig == "long" else "SELL"
        print(f"  [MR/{name}] segnale {sig.upper()} — size={size:.2f} "
              f"stop={stop_distance_pts:.1f}pt target={limit_distance_pts:.1f}pt "
              f"margine_richiesto={margin_required:.2f} EUR (dry_run={DRY_RUN})")

        result = session.place_order(
            instrument=name, direction=direction, size=size,
            stop_distance=stop_distance_pts, limit_distance=limit_distance_pts, dry_run=DRY_RUN,
        )

        if DRY_RUN:
            record_candidate_mr(name, signals, last_closed_idx, now, was_executed=0,
                                 skip_reason="dry_run")
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
            f"UPDATE live_daily_state SET orders_today_mr = orders_today_mr + 1, "
            f"capital_current_mr = {capital} WHERE trade_date = '{today_str}'"
        )
        candidate_key = record_candidate_mr(name, signals, last_closed_idx, now, was_executed=1,
                                             skip_reason=None)
        print(f"    Posizione MR aperta su IG, deal_id={deal_id}, candidate_key={candidate_key}")

        if margine_libero is not None:
            margine_libero -= margin_required

    return margine_libero


def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN mancanti.")
        sys.exit(1)

    print(f"=== live_execute.py — DRY_RUN={DRY_RUN} — {datetime.now(timezone.utc).isoformat()} ===\n")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    creds = load_credentials_from_env()
    with IGSession(creds) as session:
        print("--- 1) Scarico storico (riusato da gestione posizioni + entrambe le strategie) ---")
        # RIORDINATO 21/07/2026: prima era dopo la gestione posizioni — il
        # log bar-per-bar (dentro manage_open_positions) ha bisogno degli
        # indicatori calcolati su hist_cache, quindi va scaricato prima.
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        warmup_start = day_start - timedelta(days=WARMUP_DAYS)
        hist_cache = {}
        for name, const in SYMBOLS.items():
            hist_cache[name] = fetch_historical(const, warmup_start, now + timedelta(minutes=30))

        print("\n--- 2) Gestione posizioni aperte (V6 + MR) + log bar-per-bar ---")
        manage_open_positions(session, today_str, hist_cache)

        print("\n--- 2b) Verifica kill switch giornaliero (separato per pool) ---")
        check_and_apply_kill_switches(session, today_str)

        print("\n--- 2c) Tetto equity reale (Livello 1, protezione accantonato) ---")
        apply_equity_cap(session, today_str)

        print("\n--- 2d) Margine reale disponibile (Livello 3, nuovo 21/07/2026) ---")
        margine_libero = compute_margin_state(session)

        print("\n--- 2e) Campionamento spread (riusa sessione, nuovo 21/07/2026) ---")
        sample_spread(session)

        print("\n--- 3) Rilevazione nuovi segnali V6 ---")
        margine_libero = detect_and_open_signals_v6(session, today_str, hist_cache, margine_libero)

        print("\n--- 4) Rilevazione nuovi segnali mean-reversion (RSI) ---")
        margine_libero = detect_and_open_signals_mr(session, today_str, hist_cache, margine_libero)

        print("\n--- 5) Log barre candidati in tracking (V6 + MR) ---")
        log_candidate_bars("v6", hist_cache, now)
        log_candidate_bars("mr", hist_cache, now)

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
