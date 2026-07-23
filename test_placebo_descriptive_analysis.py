"""
test_placebo_descriptive_analysis.py — Terzo controllo di calibrazione:
verifica se l'analisi SQL descrittiva a quadranti (usata tutto il
giorno per generare ipotesi) produce pattern "interessanti" anche su
dati MESCOLATI a caso, dove per costruzione non c'e' nessuna relazione
reale tra regime e esito.

METODO: prendo i trade reali (R-multiple/win gia' avvenuti, MAI
modificati) e le loro condizioni di regime reali (atr_pct,
corr_dax_ftse_7d all'entry, gia' osservate). Poi MESCOLO a caso quali
condizioni di regime sono associate a quali trade (permutazione — rompe
qualunque relazione temporale/causale reale, ma preserva la
distribuzione marginale di entrambe le variabili). Ricalcolo lo stesso
identico quadrante (nessuno/solo_atr_alto/solo_corr_alta/stress ->
win rate) su ogni permutazione, muliple volte (N_PERMUTATIONS), per
costruire una distribuzione nulla di "quanto spread puo' produrre il
puro caso con questo campione". Poi confronto lo spread REALE (dati
non mescolati) contro questa distribuzione — se lo spread reale cade
ben oltre il 95-99esimo percentile del rumore, e' un'evidenza che il
pattern osservato oggi non era solo fortuna campionaria.

METRICA: spread = max(win_rate quadrante) - min(win_rate quadrante),
sui 4 quadranti (nessuno, solo_atr_alto, solo_corr_alta, stress),
calcolato separatamente per DAX e FTSE100 — stessa definizione usata
nell'analisi originale di oggi (DAX: 41.5%->34.2%=7.3pt osservato,
FTSE100: 39.4%->30.0%=9.4pt osservato).

Nessuna scrittura su D1.
"""
import os
import requests
import numpy as np
import pandas as pd

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

N_PERMUTATIONS = 1000
SEED = 42

# Soglie terzile gia' usate in tutta la sessione di oggi (NTILE reale su
# market_regime_indicators, non ricalcolate qui)
ATR_THRESH = {"DAX": 0.24747954706907585, "FTSE100": 0.2031323100223204}
CORR_THRESH = 0.7853464827260775


def d1_query_paginated(sql_base, chunk=5000):
    rows = []
    offset = 0
    while True:
        sql = f"{sql_base} LIMIT {chunk} OFFSET {offset}"
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
        headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
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


def quadrant_label(atr_alto: bool, corr_alta: bool) -> str:
    if atr_alto and corr_alta:
        return "stress"
    if atr_alto and not corr_alta:
        return "solo_atr_alto"
    if not atr_alto and corr_alta:
        return "solo_corr_alta"
    return "nessuno"


def compute_spread(df: pd.DataFrame) -> tuple[float, dict]:
    """df ha colonne atr_pct, corr, win (0/1). Ritorna (spread, dettaglio per quadrante)."""
    win_rates = {}
    for label, group in df.groupby("quadrante"):
        if len(group) >= 5:  # soglia minima campione per cella, evita rumore su n troppo piccoli
            win_rates[label] = group["win"].mean() * 100
    if len(win_rates) < 2:
        return 0.0, win_rates
    spread = max(win_rates.values()) - min(win_rates.values())
    return spread, win_rates


def main():
    print("Scarico trade DAX+FTSE100 con regime reale all'entry...")
    rows = d1_query_paginated(f"""
        SELECT t.instrument, t.r_multiple,
               m.atr_pct, m.corr_dax_ftse_7d
        FROM research_v6_trade_features_continuous t
        LEFT JOIN market_regime_indicators m
          ON m.instrument = t.instrument AND m.timestamp = t.entry_time
        WHERE t.exit_reason IN ('take_profit','stop_loss')
          AND m.atr_pct IS NOT NULL AND m.corr_dax_ftse_7d IS NOT NULL
    """)
    df_all = pd.DataFrame(rows)
    df_all["win"] = (df_all["r_multiple"] > 0).astype(int)
    print(f"  {len(df_all)} trade totali con regime noto.\n")

    rng = np.random.default_rng(SEED)

    for inst_name in ("DAX", "FTSE100"):
        df = df_all[df_all["instrument"] == inst_name].reset_index(drop=True)
        atr_thresh = ATR_THRESH[inst_name]

        atr_alto = df["atr_pct"] > atr_thresh
        corr_alta = df["corr_dax_ftse_7d"] > CORR_THRESH
        df["quadrante"] = [quadrant_label(a, c) for a, c in zip(atr_alto, corr_alta)]

        real_spread, real_detail = compute_spread(df)
        print(f"=== {inst_name} (n={len(df)}) ===")
        print(f"  Spread REALE (dati non mescolati): {real_spread:.1f}pt")
        for label, wr in sorted(real_detail.items()):
            print(f"    {label:16s}: {wr:.1f}%")

        # Permutazioni: mescolo (atr_pct, corr_dax_ftse_7d) in blocco
        # (mantengo la coppia intatta, mescolo solo l'abbinamento con i
        # trade) — rompe la relazione reale, preserva le distribuzioni
        # marginali di regime e di win rate separatamente.
        null_spreads = np.empty(N_PERMUTATIONS)
        atr_vals = df["atr_pct"].values
        corr_vals = df["corr_dax_ftse_7d"].values
        win_vals = df["win"].values
        n = len(df)

        for p in range(N_PERMUTATIONS):
            perm_idx = rng.permutation(n)
            atr_p = atr_vals[perm_idx]
            corr_p = corr_vals[perm_idx]
            quad_p = [quadrant_label(a > atr_thresh, c > CORR_THRESH) for a, c in zip(atr_p, corr_p)]
            df_perm = pd.DataFrame({"quadrante": quad_p, "win": win_vals})
            s, _ = compute_spread(df_perm)
            null_spreads[p] = s

        pct_rank = (null_spreads < real_spread).mean() * 100
        pct_ge = (null_spreads >= real_spread).mean() * 100

        print(f"\n  Distribuzione nulla (rumore puro, {N_PERMUTATIONS} permutazioni):")
        print(f"    media={null_spreads.mean():.1f}pt  mediana={np.median(null_spreads):.1f}pt  "
              f"p95={np.percentile(null_spreads, 95):.1f}pt  p99={np.percentile(null_spreads, 99):.1f}pt  "
              f"max={null_spreads.max():.1f}pt")
        print(f"    Spread reale ({real_spread:.1f}pt) e' al percentile {pct_rank:.1f}% della distribuzione nulla")
        print(f"    Percentuale di permutazioni casuali con spread >= reale: {pct_ge:.1f}%")

        if pct_ge <= 5:
            print(f"    -> Il pattern reale supera il 95% delle permutazioni casuali — "
                  f"difficilmente spiegabile da puro rumore campionario.")
        elif pct_ge <= 20:
            print(f"    -> Il pattern reale e' nella parte alta della distribuzione nulla ma non estremo — "
                  f"cautela, potrebbe essere in parte rumore.")
        else:
            print(f"    -> Il pattern reale e' DENTRO il range normale del rumore puro — "
                  f"un pattern casuale di questa taglia non sarebbe stato sorprendente.")
        print()

    print("=" * 90)
    print("Nota: questo calibra la soglia intuitiva usata oggi per giudicare un pattern "
          "'interessante abbastanza da testare causalmente' — non sostituisce il test "
          "causale vero (motore+bootstrap+holdout), che resta l'unico giudice finale.")


if __name__ == "__main__":
    main()
