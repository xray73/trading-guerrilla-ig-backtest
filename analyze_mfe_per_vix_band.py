"""
analyze_mfe_per_vix_band.py — Test diagnostico (19/07/2026), PRIMA di
costruire un vero motore/filtro: la modulazione di target/stop in base
al VIX avrebbe senso? Misura l'MFE (Maximum Favorable Excursion — di
quanto il prezzo si è mosso a favore, in multipli di R, PRIMA di
chiudersi) per ciascun trade del contesto V6 (ADX>20 + breakout +
trend ampio confermato), diviso per fascia VIX del giorno.

LOGICA DEL TEST: se le fasce VIX deboli hanno MFE tipicamente basso
(i trade non si avvicinano mai al target prima di girarsi), un target
più stretto in quella fascia NON aiuterebbe — il problema è la
qualità del segnale, non la calibrazione dell'uscita (coerente con
RCA: 8 varianti di gestione uscita già fallite). Se invece hanno MFE
alto (si avvicinano al target ma poi si girano comunque, per via
dello stop attuale troppo largo o del target troppo lontano), una
modulazione target/stop specifica per fascia potrebbe avere una base
reale.

ATTENZIONE ESPLICITA: questo territorio (modulazione uscita + regime
esterno) è l'intersezione di due categorie con precedenti negativi nel
progetto (8/8 varianti di uscita fallite, 3/3 tentativi di regime ATR
falliti) — il VIX è concettualmente nuovo (variabile esterna, non
derivata dal prezzo) su un campione più ampio di quanto disponibile
prima, ma il precedente resta un campanello d'allarme da rispettare.
Questo è un test ESPLORATIVO, non ancora un filtro da adottare.

Output SOLO aggregato (percentili MFE per fascia) — mai trade singoli.

Dati: adx_diagnostic_raw (D1) + VIX storico giornaliero (Yahoo
Finance). Nessuna scrittura su D1.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

ATR_STOP_MULT = 1.5   # stop attuale del sistema, come riferimento per esprimere MFE in "R"
MAX_HOLD = 48

VIX_CALMO_SOGLIA = 15.0
VIX_PANICO_SOGLIA = 25.0


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_symbol_full(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            "SELECT bar_index, timestamp, close, high, low, atr, adx, ema20, ema50, "
            "ema100, ema200, rolling_high_20, rolling_low_20, rolling_high_40, rolling_low_40 "
            f"FROM adx_diagnostic_raw WHERE symbol='{symbol}' "
            f"ORDER BY bar_index LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("bar_index").reset_index(drop=True)


def compute_mfe_and_outcome(df: pd.DataFrame, entry_pos: int, direction: str) -> tuple[float, str]:
    """Ritorna (MFE in multipli di R, esito con stop/target attuali 1.5/3.0xATR).
    R = ATR_STOP_MULT * atr (l'unità di rischio, coerente col resto del progetto)."""
    row = df.iloc[entry_pos]
    entry_price = row["close"]
    atr = row["atr"]
    r_unit = ATR_STOP_MULT * atr

    if direction == "long":
        stop = entry_price - r_unit
        target = entry_price + 2 * r_unit  # target attuale = 2R oltre lo stop
    else:
        stop = entry_price + r_unit
        target = entry_price - 2 * r_unit

    end_pos = min(entry_pos + MAX_HOLD, len(df) - 1)
    mfe_r = 0.0
    esito = "TIMEOUT"

    for j in range(entry_pos + 1, end_pos + 1):
        bar = df.iloc[j]
        if direction == "long":
            favorable_move = (bar["high"] - entry_price) / r_unit
            mfe_r = max(mfe_r, favorable_move)
            if bar["low"] <= stop:
                esito = "STOP"
                break
            if bar["high"] >= target:
                esito = "TARGET"
                break
        else:
            favorable_move = (entry_price - bar["low"]) / r_unit
            mfe_r = max(mfe_r, favorable_move)
            if bar["high"] >= stop:
                esito = "STOP"
                break
            if bar["low"] <= target:
                esito = "TARGET"
                break

    return mfe_r, esito


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

    log("=== MFE per fascia VIX — contesto V6 completo (ADX>20 + breakout + trend ampio) ===\n")

    log("Scarico storico VIX (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix_by_date = vix.set_index("Date")["Close"]
    log(f"  {len(vix)} barre VIX caricate.\n")

    all_rows = []

    for symbol in ("DAX", "FTSE100"):
        log(f"--- {symbol} ---")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

        direction = pd.Series(np.where(df["ema20"] > df["ema50"], "long", "short"), index=df.index)
        trend_ampio_ok = np.where(direction == "long", df["ema100"] > df["ema200"], df["ema100"] < df["ema200"])

        if symbol == "DAX":
            dist_r = np.where(direction == "long",
                               (df["close"] - df["rolling_high_20"]) / df["atr"],
                               (df["rolling_low_20"] - df["close"]) / df["atr"])
        else:
            dist_r = np.where(direction == "long",
                               (df["close"] - df["rolling_high_40"]) / df["atr"],
                               (df["rolling_low_40"] - df["close"]) / df["atr"])

        mask = (df["adx"] > 20) & (dist_r >= 0) & trend_ampio_ok
        positions = df.index[mask].tolist()
        log(f"  Trade nel contesto V6 completo: {len(positions)}")

        # fascia VIX per data (calcolata una volta per data unica)
        entry_dates = df["timestamp"].dt.tz_localize(None).dt.normalize()
        unique_dates = entry_dates.unique()
        date_to_fascia = {}
        for d in unique_dates:
            vix_val = None
            for delta in range(0, 6):
                check_date = pd.Timestamp(d) - pd.Timedelta(days=delta)
                if check_date in vix_by_date.index:
                    vix_val = vix_by_date.loc[check_date]
                    break
            if vix_val is None:
                date_to_fascia[d] = None
            elif vix_val < VIX_CALMO_SOGLIA:
                date_to_fascia[d] = "calmo"
            elif vix_val > VIX_PANICO_SOGLIA:
                date_to_fascia[d] = "panico"
            else:
                date_to_fascia[d] = "medio"
        vix_fascia_all = entry_dates.map(date_to_fascia)

        log("  Calcolo MFE per ciascun trade (può richiedere qualche minuto)...")
        risultati = []
        for pos in positions:
            fascia = vix_fascia_all.iloc[pos]
            if fascia is None:
                continue
            mfe_r, esito = compute_mfe_and_outcome(df, pos, direction.iloc[pos])
            risultati.append({"fascia_vix": fascia, "mfe_r": mfe_r, "esito": esito})

        df_ris = pd.DataFrame(risultati)
        df_ris["symbol"] = symbol
        all_rows.append(df_ris)

        log(f"\n  --- Distribuzione MFE per fascia VIX ({symbol}) ---")
        for fascia in ("calmo", "medio", "panico"):
            sub = df_ris[df_ris["fascia_vix"] == fascia]
            if sub.empty:
                continue
            mfe = sub["mfe_r"]
            n_stop = (sub["esito"] == "STOP").sum()
            n_stop_con_mfe_oltre_1r = ((sub["esito"] == "STOP") & (sub["mfe_r"] >= 1.0)).sum()
            pct_stop_avvicinati = n_stop_con_mfe_oltre_1r / n_stop * 100 if n_stop > 0 else float("nan")
            log(f"    {fascia:<8} (n={len(sub):>5}): MFE mediano={mfe.median():.2f}R  "
                f"p25={mfe.quantile(0.25):.2f}R  p75={mfe.quantile(0.75):.2f}R  p90={mfe.quantile(0.90):.2f}R")
            log(f"             tra gli STOP (n={n_stop}): {pct_stop_avvicinati:.1f}% avevano raggiunto "
                f"almeno 1R di favore prima di girarsi")
        log("")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv("results/mfe_per_vix_band_riepilogo.csv", index=False)
    with open("results/analyze_mfe_per_vix_band.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
