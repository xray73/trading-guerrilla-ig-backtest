"""
check_correlation_regime.py — CONTO GREZZO (non causale): la
correlazione rolling DAX-FTSE100 (return bar-a-bar) predice la qualita'
dei trade V6? Griglia di finestre (3/5/7/14/21 giorni), stessa logica
gia' usata oggi per ATR% (istantaneo -> rolling, nessuna finestra
"magica" ha battuto la versione istantanea). Qui si parte direttamente
con una griglia, non un singolo punto.

Applicato a TUTTI i trade V6 eseguiti (DAX+FTSE100 insieme) — la
correlazione e' un regime di mercato condiviso, non specifico di un
singolo strumento (a differenza dell'ATR%, che era proprieta' del
singolo strumento).

Nessuna modifica a engine.py. Uses D1 REST via requests (necessario:
il join cross-strumento per il calcolo della correlazione va oltre i
limiti CPU delle query dirette D1 gia' documentati nel progetto).
"""
import os
import pandas as pd
import numpy as np

from ohlc_data_source import get_ohlc
import engine as eng

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

ATR_MULT = 1.5
RR_TARGET = 2.0
MAX_HOLDING_BARS = 48
WINDOWS_DAYS = [3, 5, 7, 14, 21]


def d1(sql):
    import requests
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]["results"]


def fetch_executed():
    return d1("SELECT candidate_key, instrument, direction, entry_time, atr_at_entry, close_entry "
               "FROM research_v6_candidates WHERE was_executed=1")


def fetch_path(candidate_key, instrument):
    table = "research_v6_candidate_path"
    return d1(f"SELECT bar_offset, high, low, close FROM {table} "
              f"WHERE candidate_key = '{candidate_key}' ORDER BY bar_offset ASC")


def r_multiple_for_trade(cand, path):
    entry_price = cand["close_entry"]
    atr = cand["atr_at_entry"]
    if entry_price is None or atr is None or not path:
        return None
    stop_dist = atr * ATR_MULT
    target_dist = stop_dist * RR_TARGET
    direction = cand["direction"]
    if direction == "long":
        stop_price = entry_price - stop_dist
        target_price = entry_price + target_dist
    else:
        stop_price = entry_price + stop_dist
        target_price = entry_price - target_dist

    for bar in path:
        offset = bar["bar_offset"]
        if offset == 0:
            continue
        high, low, close = bar["high"], bar["low"], bar["close"]
        if high is None or low is None:
            continue
        if direction == "long":
            if low <= stop_price:
                return -1.0
            if high >= target_price:
                return RR_TARGET
        else:
            if high >= stop_price:
                return -1.0
            if low <= target_price:
                return RR_TARGET
        if offset >= MAX_HOLDING_BARS:
            return (close - entry_price) / stop_dist if direction == "long" else (entry_price - close) / stop_dist
    last = path[-1]
    return ((last["close"] - entry_price) / stop_dist if direction == "long"
            else (entry_price - last["close"]) / stop_dist)


def main():
    print("Scarico OHLC continuo DAX+FTSE100...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Calcolo return bar-a-bar e allineo le due serie sullo stesso timestamp...")
    returns = {}
    for name, df in hist.items():
        s = df.set_index("timestamp")["close"].pct_change()
        returns[name] = s
    aligned = pd.concat([returns["DAX"].rename("dax"), returns["FTSE100"].rename("ftse")], axis=1).dropna()
    print(f"  {len(aligned)} barre allineate")

    print("Scarico trade V6 eseguiti (DAX+FTSE100)...")
    executed = fetch_executed()
    print(f"  {len(executed)} trade eseguiti")

    print("Ricostruisco R-multiple per ogni trade (rigioco path)...")
    trade_records = []
    for cand in executed:
        path = fetch_path(cand["candidate_key"], cand["instrument"])
        r = r_multiple_for_trade(cand, path)
        if r is not None:
            trade_records.append({
                "candidate_key": cand["candidate_key"],
                "entry_time": pd.Timestamp(cand["entry_time"]),
                "r": r,
            })
    trades_df = pd.DataFrame(trade_records).sort_values("entry_time").reset_index(drop=True)
    print(f"  {len(trades_df)} trade con R-multiple calcolato")

    print(f"\n=== GRIGLIA CORRELAZIONE ROLLING (finestre: {WINDOWS_DAYS} giorni) ===")
    for window_days in WINDOWS_DAYS:
        window = f"{window_days}D"
        rolling_corr = aligned["dax"].rolling(window).corr(aligned["ftse"])
        rolling_corr = rolling_corr.dropna()

        # merge_asof: correlazione piu' recente DISPONIBILE PRIMA dell'entry_time del trade
        corr_df = rolling_corr.rename("corr").to_frame().reset_index().rename(columns={"timestamp": "corr_time"})
        merged = pd.merge_asof(trades_df.sort_values("entry_time"), corr_df.sort_values("corr_time"),
                                left_on="entry_time", right_on="corr_time", direction="backward")
        merged = merged.dropna(subset=["corr"])

        median_corr = merged["corr"].median()
        high = merged[merged["corr"] >= median_corr]
        low = merged[merged["corr"] < median_corr]

        print(f"\n  --- Finestra {window_days} giorni --- (mediana corr={median_corr:.3f})")
        print(f"  Corr ALTA (>=mediana): n={len(high)} avg_r={high['r'].mean():.3f} sum_r={high['r'].sum():.2f}")
        print(f"  Corr BASSA (<mediana): n={len(low)} avg_r={low['r'].mean():.3f} sum_r={low['r'].sum():.2f}")

        # quartili per un quadro piu' fine
        try:
            merged["quartile"] = pd.qcut(merged["corr"], 4, labels=["Q1(bassa)", "Q2", "Q3", "Q4(alta)"])
            q_stats = merged.groupby("quartile")["r"].agg(["count", "mean", "sum"])
            print(f"  Per quartile:")
            for q, row in q_stats.iterrows():
                print(f"    {q}: n={int(row['count'])} avg_r={row['mean']:.3f} sum_r={row['sum']:.2f}")
        except ValueError:
            print("  (troppi pochi valori distinti per i quartili in questa finestra)")


if __name__ == "__main__":
    main()
