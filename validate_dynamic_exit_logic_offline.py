"""
validate_dynamic_exit_logic_offline.py — Simula la regola di uscita
dinamica (early_slope congelato, slope_locale, streak di conferma,
Ramo A/B) BAR PER BAR sui path già estratti in
research_v6_trade_path_continuous, replicando esattamente la logica di
BacktestEngineDynamicExitCombined._apply_dynamic_rule SENZA dover
rilanciare motore+bootstrap ogni volta.

PERCHÉ SERVE (lezione da v3, 23/07/2026): la validazione precedente
controllava la capacità discriminante SOLO a un istante fisso (barra
4), ma il motore controlla la condizione OGNI barra per tutta la
durata del trade — una metrica più reattiva a un singolo controllo può
comportarsi peggio se controllata di continuo, perché ha più occasioni
di scattare per rumore locale su un trade lungo. Questa simulazione
replica il comportamento CONTINUO, stessa granularità del motore vero.

COSA CATTURA: l'esito per-trade della regola (Ramo A: R di uscita
anticipata; Ramo B: se/quando il prezzo tocca lo stop bloccato) sugli
STESSI trade già avvenuti nel baseline — cioè l'effetto principale
(vincenti tagliati, perdenti salvati) trovato oggi essere dominante.

COSA NON CATTURA: l'effetto a cascata di portafoglio (uno slot
liberato prima che fa scattare un trade NUOVO, mai visto nel
baseline) — quello richiede comunque il motore vero. Ma dato che oggi
l'effetto dominante e' risultato essere "vincenti tagliati", non il
rientro a cascata, questa simulazione dovrebbe predire la direzione e
la magnitudine approssimativa del risultato del motore vero con buona
affidabilita', permettendo di scartare rapidamente le varianti che non
migliorano PRIMA di investire in un altro giro motore+bootstrap.

Nessuna scrittura su D1. Stampa solo aggregati.
"""
import os
import requests
import pandas as pd
import numpy as np

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

# --- Parametri della regola, stessi nomi/valori del motore ---
NEG_THRESHOLD = -0.2
POS_THRESHOLD = 0.3
LOCK_FRACTION = 0.5
DECEL_RATIO = 0.3
MIN_BARS_EARLY = 3
CONFIRM_BARS = 2
SLOPE_MODE = "locale"  # "locale" (ultime 2 barre) oppure "cumulata" (media da inizio trade)


def d1_query_paginated(sql_base, account_id, token, chunk=5000):
    rows = []
    offset = 0
    while True:
        sql = f"{sql_base} LIMIT {chunk} OFFSET {offset}"
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{D1_ID}/query"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(data.get("errors"))
        batch = data["result"][0]["results"]
        if not batch:
            break
        rows.extend(batch)
        offset += chunk
        if len(batch) < chunk:
            break
    return rows


def slope_ols(ys):
    n = len(ys)
    xs = list(range(n))
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def simulate_trade(path_df, exit_reason_baseline):
    adx_hist = []
    early_slope = None
    neg_streak = 0
    pos_streak = 0
    locked_r = None

    n_bars = len(path_df)
    for i in range(n_bars):
        bar_offset = int(path_df.iloc[i]["bar_offset"])
        adx_now = path_df.iloc[i]["adx"]
        price_r = path_df.iloc[i]["price_r"]
        if pd.isna(adx_now):
            continue
        adx_hist.append(adx_now)

        if bar_offset == MIN_BARS_EARLY - 1:
            early_slope = slope_ols(adx_hist)

        if bar_offset < MIN_BARS_EARLY or early_slope is None or early_slope == 0:
            continue

        if SLOPE_MODE == "locale":
            if len(adx_hist) < 3:
                continue
            slope_now = (adx_hist[-1] - adx_hist[-3]) / 2.0
        else:
            slope_now = (adx_hist[-1] - adx_hist[0]) / bar_offset

        is_decel = slope_now < early_slope * DECEL_RATIO
        if not is_decel:
            neg_streak = 0
            pos_streak = 0
            continue

        if locked_r is not None and price_r <= locked_r:
            return {"esito": "ramoB_stop_toccato", "bar": bar_offset, "r_finale": locked_r}

        if price_r <= NEG_THRESHOLD:
            neg_streak += 1
            pos_streak = 0
        elif price_r >= POS_THRESHOLD:
            pos_streak += 1
            neg_streak = 0
        else:
            neg_streak = 0
            pos_streak = 0

        if neg_streak >= CONFIRM_BARS:
            return {"esito": "ramoA_chiusura", "bar": bar_offset, "r_finale": price_r}

        if pos_streak >= CONFIRM_BARS:
            target_locked = price_r * LOCK_FRACTION
            if locked_r is None or target_locked > locked_r:
                locked_r = target_locked

    final_r = path_df.iloc[-1]["price_r"]
    if locked_r is not None and final_r < locked_r:
        return {"esito": "ramoB_stop_toccato_fine", "bar": n_bars - 1, "r_finale": locked_r}
    return {"esito": "invariato", "bar": n_bars - 1, "r_finale": final_r}


def main():
    print(f"Modalità pendenza: {SLOPE_MODE} | CONFIRM_BARS={CONFIRM_BARS} | "
          f"NEG={NEG_THRESHOLD} POS={POS_THRESHOLD} LOCK={LOCK_FRACTION} DECEL_RATIO={DECEL_RATIO}\n")

    print("Scarico trade features (take_profit/stop_loss, hold_bars>=6)...")
    feat_rows = d1_query_paginated(
        "SELECT trade_key, instrument, exit_reason, hold_bars, r_multiple FROM "
        "research_v6_trade_features_continuous WHERE exit_reason IN ('take_profit','stop_loss') "
        "AND hold_bars >= 6 ORDER BY trade_key",
        CF_ACCOUNT_ID, CF_API_TOKEN)
    feat_df = pd.DataFrame(feat_rows)
    print(f"  {len(feat_df)} trade.")

    print("Scarico path (adx, price_r) per questi trade...")
    path_rows = d1_query_paginated(
        "SELECT trade_key, bar_offset, adx, price_r FROM research_v6_trade_path_continuous "
        "ORDER BY trade_key, bar_offset",
        CF_ACCOUNT_ID, CF_API_TOKEN)
    path_df = pd.DataFrame(path_rows)
    print(f"  {len(path_df)} righe di path.\n")

    path_by_trade = {k: v.reset_index(drop=True) for k, v in path_df.groupby("trade_key")}

    results = []
    for _, row in feat_df.iterrows():
        tk = row["trade_key"]
        if tk not in path_by_trade:
            continue
        sim = simulate_trade(path_by_trade[tk], row["exit_reason"])
        results.append({
            "trade_key": tk, "instrument": row["instrument"], "exit_reason_baseline": row["exit_reason"],
            "r_baseline": row["r_multiple"], **sim,
        })

    res_df = pd.DataFrame(results)

    print("=== RIEPILOGO PER exit_reason_baseline x esito simulato ===")
    summary = res_df.groupby(["exit_reason_baseline", "esito"]).agg(
        n=("trade_key", "count"), r_baseline_medio=("r_baseline", "mean"), r_finale_medio=("r_finale", "mean")
    ).reset_index()
    print(summary.to_string(index=False))

    res_df["delta_r"] = res_df["r_finale"] - res_df["r_baseline"]
    delta_totale_r = res_df["delta_r"].sum()
    n_toccati = (res_df["esito"] != "invariato").sum()
    n_totali = len(res_df)

    print(f"\n=== STIMA AGGREGATA (proxy in R, non EUR — non cattura effetto a cascata) ===")
    print(f"Trade totali analizzati: {n_totali}")
    print(f"Trade toccati da un ramo: {n_toccati} ({100*n_toccati/n_totali:.1f}%)")
    print(f"Delta R totale stimato: {delta_totale_r:+.2f}")
    print(f"Delta R medio per trade: {delta_totale_r/n_totali:+.4f}")

    for inst in res_df["instrument"].unique():
        sub = res_df[res_df["instrument"] == inst]
        print(f"\n  {inst}: delta R totale={sub['delta_r'].sum():+.2f}  "
              f"(n toccati={int((sub['esito']!='invariato').sum())}/{len(sub)})")

    print("\n=== CONFRONTO CON MOTORE VERO (per calibrare l'affidabilità di questo proxy) ===")
    print("v3 motore vero (slope_locale, CONFIRM_BARS=2): z=-3.22 aggregato, delta=-10786.91 EUR")
    print("Se il segno e l'ordine di grandezza relativo qui sotto sono coerenti con quello, "
          "il proxy e' affidabile per iterare rapidamente su nuove varianti prima del motore vero.")


if __name__ == "__main__":
    main()
