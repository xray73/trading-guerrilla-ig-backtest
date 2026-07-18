"""
engine_orario_filtro_test.py — Sanity check obbligatorio + test di
impatto sui 5 periodi ufficiali per BacktestEngineOrarioFiltro
(engine_orario_filtro.py).

CHECK A — equivalenza a parametri neutri (protocollo standard del
progetto, già usato per ogni altra sottoclasse del motore): con
blocked_hours=set() (nessuna ora bloccata), il motore filtrato deve
produrre risultati IDENTICI al motore standard
(BacktestEngineFloatingKillSwitch).

CHECK B — impatto sui 5 periodi ufficiali: confronta baseline (motore
standard) contro filtro attivo (blocca 20-23 UTC, stessa fascia per
DAX e FTSE100 — decisione esplicita 18/07/2026), capitale reale 2.000€.

CRITERIO DI PROMOZIONE, fissato PRIMA di vedere i risultati
(18/07/2026): il filtro viene adottato solo se migliora il rapporto
PnL/|drawdown| su ALMENO 4 PERIODI SU 5, senza far scendere il numero
di trade sotto la soglia di significatività (30) in nessun periodo.

Dati: D1 (ohlc_prices, 5 periodi ufficiali già caricati). Nessuna
scrittura su D1. Solo stampa a log + file risultati/ per l'artifact.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import requests

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_orario_filtro import BacktestEngineOrarioFiltro, DEFAULT_BLOCKED_HOURS_UTC

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

CAPITAL0 = 2000.0  # vincolo di capitale reale

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-13"),
}
WARMUP_DAYS = 250  # copre EMA200 con margine
MIN_TRADES_SIGNIFICATIVI = 30


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_range(symbol: str, start: str, end: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' AND timeframe='30m' "
            f"AND timestamp >= '{start}' AND timestamp < '{end}' "
            f"ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    os.makedirs("results", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Sanity check + impatto: filtro orario (blocca {sorted(DEFAULT_BLOCKED_HOURS_UTC)} UTC) ===\n")

    # ================================================================
    # CHECK A — equivalenza a parametri neutri (usa un solo periodo, 2023)
    # ================================================================
    log("--- CHECK A: equivalenza a parametri neutri (blocked_hours vuoto) ---")
    start_str, end_str = PERIODS["2023"]
    warm_start = (pd.Timestamp(start_str) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    data_check = {}
    for name in ("DAX", "FTSE100"):
        df = fetch_range(name, warm_start, end_str, account_id, token)
        inst = eng.INSTRUMENTS[name]
        signals = eng.generate_signals(df, inst)
        data_check[name] = signals[signals["timestamp"] >= pd.Timestamp(start_str, tz="UTC")].reset_index(drop=True)

    base_engine = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    base_trades, _ = base_engine.run(data_check)

    filtro_engine = BacktestEngineOrarioFiltro(capital0=CAPITAL0, blocked_hours=set())
    filtro_trades, _ = filtro_engine.run(data_check)

    check_a_pass = len(base_trades) == len(filtro_trades) and filtro_engine.n_blocked_by_hour == 0
    if check_a_pass and not base_trades.empty:
        cols = ["instrument", "direction", "entry_time", "exit_time", "pnl"]
        check_a_pass = base_trades[cols].reset_index(drop=True).equals(filtro_trades[cols].reset_index(drop=True))

    log(f"  Trade motore base: {len(base_trades)}  |  Trade motore filtro (vuoto): {len(filtro_trades)}")
    log(f"  n_blocked_by_hour (deve essere 0): {filtro_engine.n_blocked_by_hour}")
    log(f"  >>> CHECK A: {'PASS' if check_a_pass else 'FAIL'}\n")

    if not check_a_pass:
        log("Check A fallito — interrompo, non ha senso procedere al test di impatto.")
        with open("results/engine_orario_filtro_test.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # ================================================================
    # CHECK B — impatto sui 5 periodi ufficiali
    # ================================================================
    log("=" * 70)
    log("CHECK B — Impatto sui 5 periodi ufficiali (capitale 2.000 EUR)")
    log("=" * 70)

    periodi_migliorati = 0
    periodi_validi = 0
    dettaglio_righe = []

    for period_label, (start_str, end_str) in PERIODS.items():
        warm_start = (pd.Timestamp(start_str) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        data_period = {}
        for name in ("DAX", "FTSE100"):
            df = fetch_range(name, warm_start, end_str, account_id, token)
            if df.empty:
                continue
            inst = eng.INSTRUMENTS[name]
            signals = eng.generate_signals(df, inst)
            data_period[name] = signals[signals["timestamp"] >= pd.Timestamp(start_str, tz="UTC")].reset_index(drop=True)

        if not data_period:
            log(f"  [{period_label}] nessun dato, salto.")
            continue

        base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        base_trades, base_metrics = base.run(data_period)

        filt = BacktestEngineOrarioFiltro(capital0=CAPITAL0)
        filt_trades, filt_metrics = filt.run(data_period)

        n_base = len(base_trades)
        n_filt = len(filt_trades)

        bm = base_metrics.iloc[0]
        fm = filt_metrics.iloc[0]

        base_dd = abs(bm['max_drawdown_pct']) if pd.notna(bm['max_drawdown_pct']) and bm['max_drawdown_pct'] != 0 else None
        filt_dd = abs(fm['max_drawdown_pct']) if pd.notna(fm['max_drawdown_pct']) and fm['max_drawdown_pct'] != 0 else None

        ratio_base = bm['pnl_total'] / base_dd if base_dd else float('nan')
        ratio_filt = fm['pnl_total'] / filt_dd if filt_dd else float('nan')

        migliora = pd.notna(ratio_filt) and pd.notna(ratio_base) and ratio_filt > ratio_base
        significativo = n_filt >= MIN_TRADES_SIGNIFICATIVI

        if significativo:
            periodi_validi += 1
            if migliora:
                periodi_migliorati += 1

        log(f"\n  --- {period_label} ---")
        log(f"    Baseline:  n_trade={n_base}  PnL={bm['pnl_total']:+.2f}  MaxDD={bm['max_drawdown_pct']*100:.2f}%  "
            f"PnL/|DD|={ratio_base:.2f}" if pd.notna(ratio_base) else f"    Baseline:  n_trade={n_base}  PnL={bm['pnl_total']:+.2f}")
        log(f"    Filtro:    n_trade={n_filt}  PnL={fm['pnl_total']:+.2f}  MaxDD={fm['max_drawdown_pct']*100:.2f}%  "
            f"PnL/|DD|={ratio_filt:.2f}  (bloccati per ora: {filt.n_blocked_by_hour})" if pd.notna(ratio_filt) else
            f"    Filtro:    n_trade={n_filt}  PnL={fm['pnl_total']:+.2f}  (bloccati per ora: {filt.n_blocked_by_hour})")
        log(f"    Trade sotto soglia significatività (<{MIN_TRADES_SIGNIFICATIVI}): {'SI' if not significativo else 'no'}")
        log(f"    Migliora PnL/|DD|: {'SI' if migliora else 'NO'}")

        dettaglio_righe.append({
            "periodo": period_label, "n_base": n_base, "n_filtro": n_filt,
            "pnl_base": bm['pnl_total'], "pnl_filtro": fm['pnl_total'],
            "dd_base": bm['max_drawdown_pct'], "dd_filtro": fm['max_drawdown_pct'],
            "ratio_base": ratio_base, "ratio_filtro": ratio_filt,
            "migliora": migliora, "significativo": significativo,
        })

    log("\n" + "=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    log(f"Periodi con campione significativo (>={MIN_TRADES_SIGNIFICATIVI} trade): {periodi_validi}/5")
    log(f"Periodi in cui il filtro migliora PnL/|DD|: {periodi_migliorati}/5")
    log(f"Criterio di promozione fissato in anticipo: >=4/5 periodi migliorati, "
        f"nessun periodo sotto soglia significatività.")

    promosso = periodi_migliorati >= 4 and periodi_validi == 5
    log(f"\n>>> FILTRO {'PROMOSSO' if promosso else 'NON PROMOSSO'} secondo il criterio fissato.")

    pd.DataFrame(dettaglio_righe).to_csv("results/engine_orario_filtro_impatto.csv", index=False)
    with open("results/engine_orario_filtro_test.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
