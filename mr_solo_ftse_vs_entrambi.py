"""
mr_solo_ftse_vs_entrambi.py — Test di impatto (18/07/2026): mean-
reversion RSI SOLO su FTSE100 contro mean-reversion su ENTRAMBI gli
strumenti (configurazione attuale), sui 5 periodi ufficiali, capitale
600€ (il sotto-pool mean-reversion reale, 30% di 2.000€).

MOTIVAZIONE: la riverifica con metodo corretto (stop/target reali,
non persistenza a punto fisso) ha mostrato DAX in mean-reversion SOTTO
la soglia di pareggio (32,7-32,8%, confermato con RSI e Bollinger,
campioni ampi), mentre FTSE100 è sopra (34,1-34,2%). Non solo "FTSE100
leggermente meglio" — "DAX potenzialmente in perdita strutturale".

NESSUNA NUOVA SOTTOCLASSE: engine_mean_reversion.py (già validato con
sanity check dedicato) è indipendente dallo strumento — l'universo
tradabile dipende solo da quali strumenti sono nel dizionario dati
passato a .run(). Questo test confronta due CONFIGURAZIONI dello
stesso motore già esistente, non richiede un nuovo Check A di
equivalenza.

CRITERIO DI PROMOZIONE, fissato PRIMA di vedere i risultati
(18/07/2026), IDENTICO al criterio A già usato per i due filtri
precedenti (orario, ADX×ATR):
  1. NESSUN PEGGIORAMENTO GRAVE: in nessuno dei 5 periodi il drawdown
     massimo di "solo FTSE100" può peggiorare di oltre il 20% relativo
     rispetto a "entrambi".
  2. MIGLIORAMENTO AGGREGATO: la somma del PnL sui 5 periodi con "solo
     FTSE100" deve essere superiore alla somma con "entrambi".
Nessun periodo può scendere sotto la soglia di significatività (20
trade — più bassa del solito 30, perché il pool mean-reversion genera
pochi trade per costruzione, atteso e già noto dal progetto).

Dati: D1 (ohlc_prices, 5 periodi ufficiali). Nessuna scrittura su D1.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import requests

import engine as eng
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

CAPITAL0_MR = 600.0  # sotto-pool mean-reversion reale (30% di 2.000 EUR)
MR_MODE = "rsi"

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-13"),
}
WARMUP_DAYS = 60  # RSI/Bollinger hanno warmup molto più corto di EMA200
MIN_TRADES_SIGNIFICATIVI = 20


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

    log("=== MR solo FTSE100 vs MR su entrambi — 5 periodi ufficiali, capitale 600 EUR ===\n")

    peggioramento_grave_trovato = False
    pnl_totale_entrambi = 0.0
    pnl_totale_solo_ftse = 0.0
    dettaglio_righe = []

    for period_label, (start_str, end_str) in PERIODS.items():
        warm_start = (pd.Timestamp(start_str) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

        mr_signals = {}
        for name in ("DAX", "FTSE100"):
            df = fetch_range(name, warm_start, end_str, account_id, token)
            if df.empty:
                continue
            inst = eng.INSTRUMENTS[name]
            sig = generate_mean_reversion_signals(df, inst, mode=MR_MODE)
            mr_signals[name] = sig[sig["timestamp"] >= pd.Timestamp(start_str, tz="UTC")].reset_index(drop=True)

        if "FTSE100" not in mr_signals:
            log(f"  [{period_label}] nessun dato FTSE100, salto.")
            continue

        # --- configurazione ENTRAMBI (attuale) ---
        eng_entrambi = BacktestEngineMeanReversion(capital0=CAPITAL0_MR)
        trades_entrambi, metrics_entrambi = eng_entrambi.run(mr_signals)

        # --- configurazione SOLO FTSE100 ---
        eng_solo_ftse = BacktestEngineMeanReversion(capital0=CAPITAL0_MR)
        trades_solo_ftse, metrics_solo_ftse = eng_solo_ftse.run({"FTSE100": mr_signals["FTSE100"]})

        n_entrambi = len(trades_entrambi)
        n_solo_ftse = len(trades_solo_ftse)

        me = metrics_entrambi.iloc[0]
        mf = metrics_solo_ftse.iloc[0]

        dd_entrambi = abs(me['max_drawdown_pct']) if pd.notna(me['max_drawdown_pct']) and me['max_drawdown_pct'] != 0 else None
        dd_solo_ftse = abs(mf['max_drawdown_pct']) if pd.notna(mf['max_drawdown_pct']) and mf['max_drawdown_pct'] != 0 else None

        significativo = n_solo_ftse >= MIN_TRADES_SIGNIFICATIVI

        peggioramento_grave = False
        peggioramento_relativo_pct = float('nan')
        if dd_entrambi is not None and dd_solo_ftse is not None and dd_entrambi > 0:
            peggioramento_relativo_pct = (dd_solo_ftse - dd_entrambi) / dd_entrambi * 100
            peggioramento_grave = peggioramento_relativo_pct > 20.0
            if peggioramento_grave:
                peggioramento_grave_trovato = True

        pnl_totale_entrambi += me['pnl_total']
        pnl_totale_solo_ftse += mf['pnl_total']

        log(f"\n  --- {period_label} ---")
        log(f"    Entrambi (attuale):  n_trade={n_entrambi}  PnL={me['pnl_total']:+.2f}  MaxDD={me['max_drawdown_pct']*100:.2f}%  "
            f"(DAX+FTSE100 mischiati, {eng_entrambi.n_skipped_min_size} saltati per size)")
        log(f"    Solo FTSE100:        n_trade={n_solo_ftse}  PnL={mf['pnl_total']:+.2f}  MaxDD={mf['max_drawdown_pct']*100:.2f}%  "
            f"({eng_solo_ftse.n_skipped_min_size} saltati per size)")
        log(f"    Peggioramento drawdown relativo: {peggioramento_relativo_pct:+.1f}% "
            f"({'GRAVE, >20%' if peggioramento_grave else 'entro tolleranza'})" if pd.notna(peggioramento_relativo_pct) else "    Peggioramento drawdown: N/A")
        log(f"    Trade sotto soglia significatività (<{MIN_TRADES_SIGNIFICATIVI}): {'SI' if not significativo else 'no'}")

        dettaglio_righe.append({
            "periodo": period_label, "n_entrambi": n_entrambi, "n_solo_ftse": n_solo_ftse,
            "pnl_entrambi": me['pnl_total'], "pnl_solo_ftse": mf['pnl_total'],
            "dd_entrambi": me['max_drawdown_pct'], "dd_solo_ftse": mf['max_drawdown_pct'],
            "peggioramento_dd_relativo_pct": peggioramento_relativo_pct,
            "peggioramento_grave": peggioramento_grave, "significativo": significativo,
        })

    log("\n" + "=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    log(f"Peggioramento grave (>20% relativo) del drawdown in qualche periodo: "
        f"{'SI' if peggioramento_grave_trovato else 'no'}")
    log(f"PnL totale 5 periodi — entrambi: {pnl_totale_entrambi:+.2f}  solo FTSE100: {pnl_totale_solo_ftse:+.2f}  "
        f"(differenza: {pnl_totale_solo_ftse - pnl_totale_entrambi:+.2f})")
    log(f"Criterio di promozione fissato in anticipo: (1) nessun peggioramento drawdown >20% "
        f"relativo in nessun periodo, E (2) PnL aggregato migliore con solo FTSE100.")

    promosso = (not peggioramento_grave_trovato) and (pnl_totale_solo_ftse > pnl_totale_entrambi)
    log(f"\n>>> 'SOLO FTSE100' {'PROMOSSO' if promosso else 'NON PROMOSSO'} secondo il criterio fissato.")

    pd.DataFrame(dettaglio_righe).to_csv("results/mr_solo_ftse_vs_entrambi.csv", index=False)
    with open("results/mr_solo_ftse_vs_entrambi.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
