"""
analyze_vix_sizing_traintest.py — Test train/test rigoroso (19/07/2026):
la modulazione della size in base alla fascia VIX (calmo/medio/panico)
migliora il risultato FUORI CAMPIONE, o è solo un artefatto circolare
del tipo già visto (calibri sullo stesso storico che usi per
"dimostrare" il miglioramento)?

METODOLOGIA — disciplina train/test standard del progetto:
  TRAIN = 2015-01-01 -> 2024-01-01 (include 2015-2016, 2020-covid, 2023)
  TEST  = 2024-01-01 -> 2026-07-19 (2024-2025 + 2026-ytd, MAI visto
          durante la calibrazione dei moltiplicatori)

REGOLA DI CALIBRAZIONE, fissata PRIMA di vedere i risultati (nessuna
scelta a mano dei pesi, per evitare l'artefatto circolare già
dimostrato con i moltiplicatori scelti manualmente):
  moltiplicatore_fascia = win_rate_fascia(train) / win_rate_medio(train)
  clip tra 0.5x e 1.5x (evita scommesse estreme in una direzione)

Popolazione: contesto V6 VERO (ADX>20 + breakout reale + trend ampio
confermato) — non la popolazione diluita usata nel calcolo grezzo
precedente. Stessa metodologia stop/target reale (1.5xATR stop,
3xATR target, max 48 barre) di tutti i test di oggi.

VERDETTO: la size modulata (moltiplicatori calibrati SOLO su train)
batte la size flat (1x sempre) SUL TEST (2024-2026, mai visto)? Se sì,
il pattern regge fuori campione. Se no (o se il vantaggio crolla),
stesso destino del filtro maturità trend (RCA sez. 17: train vince,
test crolla) — pattern classico di overfitting.

Output SOLO aggregato — mai trade singoli elencati.

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

ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
MAX_HOLD = 48

VIX_CALMO_SOGLIA = 15.0
VIX_PANICO_SOGLIA = 25.0

TRAIN_END = "2024-01-01"
MULT_MIN, MULT_MAX = 0.5, 1.5


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


def simulate_outcome(df: pd.DataFrame, entry_pos: int, direction: str) -> str:
    row = df.iloc[entry_pos]
    entry_price = row["close"]
    atr = row["atr"]
    if direction == "long":
        stop = entry_price - ATR_STOP_MULT * atr
        target = entry_price + ATR_TARGET_MULT * atr
    else:
        stop = entry_price + ATR_STOP_MULT * atr
        target = entry_price - ATR_TARGET_MULT * atr

    end_pos = min(entry_pos + MAX_HOLD, len(df) - 1)
    for j in range(entry_pos + 1, end_pos + 1):
        bar = df.iloc[j]
        if direction == "long":
            if bar["low"] <= stop:
                return "STOP"
            if bar["high"] >= target:
                return "TARGET"
        else:
            if bar["high"] >= stop:
                return "STOP"
            if bar["low"] <= target:
                return "TARGET"
    return "TIMEOUT"


def get_v6_context_positions(df: pd.DataFrame, symbol: str) -> tuple[list[int], pd.Series]:
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
    return df.index[mask].tolist(), direction


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

    log("=== Modulazione size per fascia VIX — TRAIN 2015-2023 / TEST 2024-2026 ===\n")

    log("Scarico storico VIX (Yahoo Finance)...")
    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    vix_by_date = vix.set_index("Date")["Close"]
    log(f"  {len(vix)} barre VIX caricate.\n")

    riepilogo = []

    for symbol in ("DAX", "FTSE100"):
        log(f"{'='*70}\n{symbol}\n{'='*70}")
        df = fetch_symbol_full(symbol, account_id, token)
        log(f"  {len(df)} barre caricate.")

        positions, direction = get_v6_context_positions(df, symbol)
        log(f"  Trade nel contesto V6 vero: {len(positions)}")

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

        log("  Simulo esiti reali per ogni trade del contesto...")
        rows = []
        for pos in positions:
            fascia = vix_fascia_all.iloc[pos]
            if fascia is None:
                continue
            esito = simulate_outcome(df, pos, direction.iloc[pos])
            rows.append({
                "bar_index": df.iloc[pos]["bar_index"],
                "timestamp": df.iloc[pos]["timestamp"],
                "fascia_vix": fascia,
                "esito": esito,
            })
        trades = pd.DataFrame(rows)
        trades["r_flat"] = trades["esito"].map({"TARGET": 2.0, "STOP": -1.0, "TIMEOUT": 0.0})

        train = trades[trades["timestamp"] < pd.Timestamp(TRAIN_END, tz="UTC")]
        test = trades[trades["timestamp"] >= pd.Timestamp(TRAIN_END, tz="UTC")]
        log(f"  Train (2015-2023): {len(train)} trade  |  Test (2024-2026): {len(test)} trade\n")

        # --- calibrazione moltiplicatori SOLO su train ---
        log("  --- Calibrazione moltiplicatori (SOLO train) ---")
        wr_train_overall = (train["esito"] == "TARGET").sum() / (
            (train["esito"] == "TARGET").sum() + (train["esito"] == "STOP").sum())
        moltiplicatori = {}
        for fascia in ("calmo", "medio", "panico"):
            sub = train[train["fascia_vix"] == fascia]
            n_t = (sub["esito"] == "TARGET").sum()
            n_s = (sub["esito"] == "STOP").sum()
            wr = n_t / (n_t + n_s) if (n_t + n_s) > 0 else wr_train_overall
            mult = np.clip(wr / wr_train_overall, MULT_MIN, MULT_MAX)
            moltiplicatori[fascia] = mult
            log(f"    {fascia:<8}: win rate train={wr*100:.2f}%  "
                f"(medio train={wr_train_overall*100:.2f}%)  moltiplicatore={mult:.3f}")

        # --- applicazione sul test, MAI visto durante la calibrazione ---
        log("\n  --- Applicazione sul TEST (2024-2026, mai visto) ---")
        test = test.copy()
        test["mult"] = test["fascia_vix"].map(moltiplicatori)
        test["r_modulato"] = test["r_flat"] * test["mult"]

        r_flat_test = test["r_flat"].sum()
        r_modulato_test = test["r_modulato"].sum()
        uplift_pct = (r_modulato_test - r_flat_test) / abs(r_flat_test) * 100 if r_flat_test != 0 else float("nan")

        log(f"    R totali FLAT (size sempre 1x):      {r_flat_test:+.1f}R")
        log(f"    R totali MODULATO (pesi da train):   {r_modulato_test:+.1f}R")
        log(f"    Differenza: {uplift_pct:+.1f}%")
        log(f"    >>> Modulazione VIX {'REGGE fuori campione' if r_modulato_test > r_flat_test else 'NON regge fuori campione'} per {symbol}\n")

        riepilogo.append({
            "symbol": symbol, "n_train": len(train), "n_test": len(test),
            "mult_calmo": moltiplicatori["calmo"], "mult_medio": moltiplicatori["medio"],
            "mult_panico": moltiplicatori["panico"],
            "r_flat_test": r_flat_test, "r_modulato_test": r_modulato_test,
            "uplift_pct": uplift_pct, "regge_test": r_modulato_test > r_flat_test,
        })

    log("=" * 70)
    log("VERDETTO FINALE")
    log("=" * 70)
    for row in riepilogo:
        log(f"  {row['symbol']}: {'REGGE' if row['regge_test'] else 'NON REGGE'} fuori campione "
            f"({row['uplift_pct']:+.1f}% su {row['n_test']} trade di test)")

    pd.DataFrame(riepilogo).to_csv("results/vix_sizing_traintest_riepilogo.csv", index=False)
    with open("results/analyze_vix_sizing_traintest.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
