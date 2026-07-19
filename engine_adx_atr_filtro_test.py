"""
engine_adx_atr_filtro_test.py — Sanity check obbligatorio + test di
impatto sui 5 periodi ufficiali per BacktestEngineADXATRFiltro
(engine_adx_atr_filtro.py).

CHECK A — equivalenza a parametri neutri: con soglie irraggiungibili
(ADX>999 per entrambi gli strumenti), il motore filtrato deve produrre
risultati IDENTICI al motore standard (BacktestEngineFloatingKillSwitch).

CHECK B — impatto sui 5 periodi ufficiali: confronta baseline contro
filtro attivo (soglie di default: DAX ADX>30+ATR%>=0.25, FTSE100
ADX>40), capitale reale 2.000€.

CRITERIO DI PROMOZIONE, fissato PRIMA di vedere i risultati
(18/07/2026), AGGIORNATO rispetto al primo tentativo (filtro orario,
"migliora in >=4/5 periodi" — criticato come troppo insensibile alla
gravità del peggioramento in un singolo periodo). Nuovo criterio,
unisce due condizioni:

  1. NESSUN PEGGIORAMENTO GRAVE: in nessuno dei 5 periodi il drawdown
     massimo del filtro può peggiorare di oltre il 20% in termini
     relativi rispetto al baseline (es. da -25% a non oltre -30%).
  2. MIGLIORAMENTO AGGREGATO: la somma del PnL sui 5 periodi con il
     filtro deve essere superiore alla somma del PnL baseline.

Più tollerante di "deve migliorare in ogni singolo periodo" (evita di
scartare un filtro buono per un'oscillazione minima di rumore in un
periodo), ma più severo di "basta vincere la maggioranza dei periodi"
(blocca un filtro che nasconde un peggioramento serio dietro una
media aggregata favorevole). Nessun periodo può scendere sotto la
soglia di significatività (30 trade).

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
from engine_adx_atr_filtro import BacktestEngineADXATRFiltro

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

CAPITAL0 = 2000.0

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-13"),
}
WARMUP_DAYS = 250
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

    log("=== Sanity check + impatto: filtro ADX x ATR (regole per strumento) ===")
    log("    DAX: salta se ADX>30 E ATR%>=0.25 (combinazione)")
    log("    FTSE100: salta se ADX>40 (soglia singola, indipendente da ATR)\n")

    # ================================================================
    # CHECK A — equivalenza a parametri neutri (usa un solo periodo, 2023)
    # ================================================================
    log("--- CHECK A: equivalenza a parametri neutri (soglie irraggiungibili) ---")
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

    filtro_engine = BacktestEngineADXATRFiltro(
        capital0=CAPITAL0, dax_adx_threshold=999.0, ftse_adx_threshold=999.0)
    filtro_trades, _ = filtro_engine.run(data_check)

    check_a_pass = (len(base_trades) == len(filtro_trades)
                     and filtro_engine.n_blocked_dax == 0 and filtro_engine.n_blocked_ftse == 0)
    if check_a_pass and not base_trades.empty:
        cols = ["instrument", "direction", "entry_time", "exit_time", "pnl"]
        check_a_pass = base_trades[cols].reset_index(drop=True).equals(filtro_trades[cols].reset_index(drop=True))

    log(f"  Trade motore base: {len(base_trades)}  |  Trade motore filtro (soglie irraggiungibili): {len(filtro_trades)}")
    log(f"  n_blocked_dax + n_blocked_ftse (devono essere 0): "
        f"{filtro_engine.n_blocked_dax} + {filtro_engine.n_blocked_ftse}")
    log(f"  >>> CHECK A: {'PASS' if check_a_pass else 'FAIL'}\n")

    if not check_a_pass:
        log("Check A fallito — interrompo, non ha senso procedere al test di impatto.")
        with open("results/engine_adx_atr_filtro_test.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # ================================================================
    # CHECK B — impatto sui 5 periodi ufficiali
    # ================================================================
    log("=" * 70)
    log("CHECK B — Impatto sui 5 periodi ufficiali (capitale 2.000 EUR)")
    log("=" * 70)

    periodi_validi = 0
    peggioramento_grave_trovato = False
    pnl_totale_base = 0.0
    pnl_totale_filtro = 0.0
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

        filt = BacktestEngineADXATRFiltro(capital0=CAPITAL0)
        filt_trades, filt_metrics = filt.run(data_period)

        n_base = len(base_trades)
        n_filt = len(filt_trades)

        bm = base_metrics.iloc[0]
        fm = filt_metrics.iloc[0]

        base_dd = abs(bm['max_drawdown_pct']) if pd.notna(bm['max_drawdown_pct']) and bm['max_drawdown_pct'] != 0 else None
        filt_dd = abs(fm['max_drawdown_pct']) if pd.notna(fm['max_drawdown_pct']) and fm['max_drawdown_pct'] != 0 else None

        ratio_base = bm['pnl_total'] / base_dd if base_dd else float('nan')
        ratio_filt = fm['pnl_total'] / filt_dd if filt_dd else float('nan')

        significativo = n_filt >= MIN_TRADES_SIGNIFICATIVI
        if significativo:
            periodi_validi += 1

        # peggioramento grave: drawdown del filtro >20% peggiore (relativo) del baseline
        peggioramento_grave = False
        if base_dd is not None and filt_dd is not None and base_dd > 0:
            peggioramento_relativo_pct = (filt_dd - base_dd) / base_dd * 100
            peggioramento_grave = peggioramento_relativo_pct > 20.0
            if peggioramento_grave:
                peggioramento_grave_trovato = True
        else:
            peggioramento_relativo_pct = float('nan')

        pnl_totale_base += bm['pnl_total']
        pnl_totale_filtro += fm['pnl_total']

        log(f"\n  --- {period_label} ---")
        log(f"    Baseline:  n_trade={n_base}  PnL={bm['pnl_total']:+.2f}  MaxDD={bm['max_drawdown_pct']*100:.2f}%")
        log(f"    Filtro:    n_trade={n_filt}  PnL={fm['pnl_total']:+.2f}  MaxDD={fm['max_drawdown_pct']*100:.2f}%  "
            f"(bloccati DAX: {filt.n_blocked_dax}, FTSE100: {filt.n_blocked_ftse})")
        log(f"    Peggioramento drawdown relativo: {peggioramento_relativo_pct:+.1f}% "
            f"({'GRAVE, >20%' if peggioramento_grave else 'entro tolleranza'})" if pd.notna(peggioramento_relativo_pct) else "    Peggioramento drawdown: N/A")
        log(f"    Trade sotto soglia significatività (<{MIN_TRADES_SIGNIFICATIVI}): {'SI' if not significativo else 'no'}")

        dettaglio_righe.append({
            "periodo": period_label, "n_base": n_base, "n_filtro": n_filt,
            "pnl_base": bm['pnl_total'], "pnl_filtro": fm['pnl_total'],
            "dd_base": bm['max_drawdown_pct'], "dd_filtro": fm['max_drawdown_pct'],
            "peggioramento_dd_relativo_pct": peggioramento_relativo_pct,
            "peggioramento_grave": peggioramento_grave,
            "bloccati_dax": filt.n_blocked_dax, "bloccati_ftse": filt.n_blocked_ftse,
            "significativo": significativo,
        })

    log("\n" + "=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    log(f"Periodi con campione significativo (>={MIN_TRADES_SIGNIFICATIVI} trade): {periodi_validi}/5")
    log(f"Peggioramento grave (>20% relativo) del drawdown in qualche periodo: "
        f"{'SI' if peggioramento_grave_trovato else 'no'}")
    log(f"PnL totale 5 periodi — baseline: {pnl_totale_base:+.2f}  filtro: {pnl_totale_filtro:+.2f}  "
        f"(differenza: {pnl_totale_filtro - pnl_totale_base:+.2f})")
    log(f"Criterio di promozione fissato in anticipo: (1) nessun peggioramento drawdown >20% "
        f"relativo in nessun periodo, E (2) PnL aggregato sui 5 periodi migliore del baseline.")

    promosso = (not peggioramento_grave_trovato) and (pnl_totale_filtro > pnl_totale_base) and periodi_validi == 5
    log(f"\n>>> FILTRO {'PROMOSSO' if promosso else 'NON PROMOSSO'} secondo il criterio fissato.")

    pd.DataFrame(dettaglio_righe).to_csv("results/engine_adx_atr_filtro_impatto.csv", index=False)
    with open("results/engine_adx_atr_filtro_test.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
