"""
persist_correlation_column.py — UNA TANTUM: calcola la correlazione
rolling DAX-FTSE100 (return bar-a-bar, finestra 7 giorni, stessa
metodologia di test_filtro_correlazione.py) e la scrive come nuova
colonna `corr_7d_entry` in research_v6_candidates, per TUTTI i
candidati (eseguiti e non). Da qui in avanti l'analisi su "cosa
distingue i trade vincenti da quelli perdenti dentro il regime ad
alta correlazione" si fa interamente via query dirette D1 — nessun
altro script necessario per questo filone.

Nessuna modifica a engine.py. Nessuna modifica alla struttura esistente
di research_v6_candidates oltre all'aggiunta di questa colonna.
"""
import os
import pandas as pd
import requests

from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
CORR_WINDOW_DAYS = 7


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]


def main():
    print("Aggiungo colonna corr_7d_entry a research_v6_candidates (se non esiste)...")
    try:
        d1("ALTER TABLE research_v6_candidates ADD COLUMN corr_7d_entry REAL")
        print("  Colonna aggiunta.")
    except RuntimeError as e:
        print(f"  (probabilmente gia' esistente, procedo comunque: {e})")

    print("\nScarico OHLC continuo DAX+FTSE100...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print(f"Calcolo correlazione rolling ({CORR_WINDOW_DAYS} giorni)...")
    returns = {name: hist[name].set_index("timestamp")["close"].pct_change() for name in hist}
    aligned = pd.concat([returns["DAX"].rename("dax"), returns["FTSE100"].rename("ftse")],
                         axis=1, sort=True).dropna()
    rolling_corr = aligned["dax"].rolling(f"{CORR_WINDOW_DAYS}D").corr(aligned["ftse"]).dropna()
    corr_df = rolling_corr.rename("corr").to_frame().reset_index().rename(columns={"timestamp": "corr_time"})
    print(f"  {len(corr_df)} punti di correlazione calcolati")

    print("\nScarico TUTTI i candidati V6 (eseguiti e non)...")
    all_candidates = d1("SELECT candidate_key, entry_time FROM research_v6_candidates")["results"]
    print(f"  {len(all_candidates)} candidati totali")

    cand_df = pd.DataFrame(all_candidates)
    cand_df["entry_time"] = pd.to_datetime(cand_df["entry_time"])

    print("Allineo correlazione a ciascun candidato (merge_asof, backward)...")
    merged = pd.merge_asof(cand_df.sort_values("entry_time"), corr_df.sort_values("corr_time"),
                            left_on="entry_time", right_on="corr_time", direction="backward")
    merged = merged.dropna(subset=["corr"])
    print(f"  {len(merged)} candidati con correlazione assegnata")

    print("\nScrivo su D1 in batch da 200...")
    batch_size = 200
    rows = merged[["candidate_key", "corr"]].values.tolist()
    n_written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        case_parts = " ".join(f"WHEN '{k}' THEN {v}" for k, v in batch)
        keys_list = ",".join(f"'{k}'" for k, v in batch)
        sql = (f"UPDATE research_v6_candidates SET corr_7d_entry = CASE candidate_key "
               f"{case_parts} END WHERE candidate_key IN ({keys_list})")
        d1(sql)
        n_written += len(batch)
        if n_written % 2000 == 0 or n_written == len(rows):
            print(f"  {n_written}/{len(rows)} scritti...")

    print(f"\nCompletato: {n_written} righe aggiornate con corr_7d_entry.")


if __name__ == "__main__":
    main()
