"""
pairs_cointegration_test.py — Verifica il prerequisito statistico per
il pairs trading DAX-FTSE100 (ramo esplorativo, 17/07/2026): la coppia
è davvero cointegrata? Senza questo, l'idea non ha basi solide — lo
studio citato in chat (information ratio 0.52-1.29) usava DAX-CAC40,
non DAX-FTSE100, quindi va verificato da zero.

Test di Engle-Granger (standard per verificare cointegrazione tra due
serie di prezzo):
  1. Regressione lineare: prezzo_DAX = alfa + beta * prezzo_FTSE100 + residuo
  2. Test di stazionarietà (ADF - Augmented Dickey-Fuller) sui residui
  3. Se i residui sono stazionari (p-value ADF < 0.05), la coppia è
     cointegrata: lo spread tende a tornare verso una media, base
     statistica valida per una strategia mean-reversion sullo spread.
  4. Se NON stazionari: la coppia non è cointegrata in questo periodo,
     il pairs trading su questi due strumenti non avrebbe basi solide.

Testato su PIÙ finestre temporali (non solo una) per vedere se la
relazione è stabile nel tempo o solo un artefatto di un periodo
specifico — stessa disciplina già applicata al resto del progetto.

Richiede statsmodels (per l'ADF test) — non ancora nelle dipendenze
standard del progetto, installata solo per questo script.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

from statsmodels.tsa.stattools import adfuller, coint

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

WINDOWS = [
    ("2015-2016", "2015-01-01", "2016-12-31"),
    ("2020-covid", "2020-01-01", "2020-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("2026-ytd", "2026-01-01", "2026-07-15"),
]


def fetch_daily_close(symbol_const, start: datetime, end: datetime) -> pd.Series:
    """Usa barre giornaliere (non 30min) — la cointegrazione si valuta
    su una scala temporale più lunga, coerente con l'orizzonte di un
    trade pairs (giorni, non minuti)."""
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_DAY_1, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df["close"]


def test_window(label: str, start: str, end: str) -> dict:
    dax = fetch_daily_close(SYMBOLS["DAX"], pd.Timestamp(start), pd.Timestamp(end))
    ftse = fetch_daily_close(SYMBOLS["FTSE100"], pd.Timestamp(start), pd.Timestamp(end))

    combined = pd.DataFrame({"DAX": dax, "FTSE100": ftse}).dropna()
    n = len(combined)
    if n < 60:
        return {"periodo": label, "n_giorni": n, "errore": "campione troppo piccolo"}

    # correlazione semplice (informativa, non sufficiente da sola per dire "cointegrata")
    correlation = combined["DAX"].corr(combined["FTSE100"])

    # test di Engle-Granger (statsmodels.tsa.stattools.coint fa già
    # regressione + ADF sui residui in un solo passo)
    score, pvalue, _ = coint(combined["DAX"], combined["FTSE100"])

    # regressione manuale per ricavare l'hedge ratio (beta) e lo spread,
    # utile per capire la relazione anche se non serve al test in sé
    beta = np.polyfit(combined["FTSE100"], combined["DAX"], 1)[0]
    spread = combined["DAX"] - beta * combined["FTSE100"]
    spread_mean = spread.mean()
    spread_std = spread.std()

    cointegrata = pvalue < 0.05

    return {
        "periodo": label, "n_giorni": n, "correlazione": correlation,
        "coint_pvalue": pvalue, "cointegrata_p<0.05": cointegrata,
        "hedge_ratio_beta": beta, "spread_medio": spread_mean, "spread_std": spread_std,
    }


def main():
    print("=== Test di cointegrazione DAX-FTSE100 (Engle-Granger) ===\n")
    print("Prerequisito per la specifica pairs trading — se la coppia non è")
    print("cointegrata in modo stabile, l'idea non ha basi solide.\n")

    results = []
    for label, start, end in WINDOWS:
        print(f"--- {label} ---")
        r = test_window(label, start, end)
        if "errore" in r:
            print(f"  {r['errore']}")
        else:
            print(f"  N giorni: {r['n_giorni']}")
            print(f"  Correlazione: {r['correlazione']:.3f}")
            print(f"  Cointegrazione (Engle-Granger) p-value: {r['coint_pvalue']:.4f} "
                  f"-> {'COINTEGRATA' if r['cointegrata_p<0.05'] else 'NON cointegrata'} (soglia 0.05)")
            print(f"  Hedge ratio (beta): {r['hedge_ratio_beta']:.4f}")
            print(f"  Spread medio: {r['spread_medio']:.1f}  (std: {r['spread_std']:.1f})\n")
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv("pairs_cointegration_results.csv", index=False)

    n_ok = sum(1 for r in results if r.get("cointegrata_p<0.05"))
    n_tot = sum(1 for r in results if "errore" not in r)
    print(f"=== RIEPILOGO: cointegrata in {n_ok}/{n_tot} periodi testati ===")
    if n_ok == n_tot:
        print("Cointegrazione stabile su tutti i periodi — base statistica solida per procedere.")
    elif n_ok == 0:
        print("MAI cointegrata in nessun periodo — la coppia DAX-FTSE100 non è adatta al pairs "
              "trading classico. Da considerare: CAC40 come contropartita del DAX invece di FTSE100.")
    else:
        print("Cointegrazione INSTABILE nel tempo — presente in alcuni periodi, assente in altri. "
              "Rischio analogo a quello già visto con ORB+ADX: nessuna base solida e stabile.")

    print("\nFile: pairs_cointegration_results.csv")


if __name__ == "__main__":
    main()
