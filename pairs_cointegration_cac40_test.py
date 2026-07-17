"""
pairs_cointegration_test.py — Verifica il prerequisito statistico per
il pairs trading DAX-CAC40 (ramo esplorativo, 17/07/2026): la coppia
è davvero cointegrata? DAX-FTSE100 è stata testata il 17/07/2026 e
NON è risultata cointegrata su nessuno dei 5 periodi ufficiali (0/5).
Questo test usa CAC40 al posto di FTSE100 — è la coppia effettivamente
validata dallo studio citato in chat (information ratio 0.52-1.29).

Costante Dukascopy verificata via discover_dukascopy_cac40.py:
INSTRUMENT_IDX_EUROPE_E_CAAC_40.

Test di Engle-Granger (standard per verificare cointegrazione tra due
serie di prezzo):
  1. Regressione lineare: prezzo_DAX = alfa + beta * prezzo_CAC40 + residuo
  2. Test di stazionarietà (ADF - Augmented Dickey-Fuller) sui residui
  3. Se i residui sono stazionari (p-value ADF < 0.05), la coppia è
     cointegrata: lo spread tende a tornare verso una media, base
     statistica valida per una strategia mean-reversion sullo spread.
  4. Se NON stazionari: la coppia non è cointegrata in questo periodo.

Testato su PIÙ finestre temporali (non solo una) per vedere se la
relazione è stabile nel tempo o solo un artefatto di un periodo
specifico — stessa disciplina già applicata al resto del progetto.

Richiede statsmodels (per l'ADF test).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_CAAC_40

from statsmodels.tsa.stattools import adfuller, coint

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "CAC40": INSTRUMENT_IDX_EUROPE_E_CAAC_40}

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
    cac40 = fetch_daily_close(SYMBOLS["CAC40"], pd.Timestamp(start), pd.Timestamp(end))

    combined = pd.DataFrame({"DAX": dax, "CAC40": cac40}).dropna()
    n = len(combined)
    if n < 60:
        return {"periodo": label, "n_giorni": n, "errore": "campione troppo piccolo"}

    # correlazione semplice (informativa, non sufficiente da sola per dire "cointegrata")
    correlation = combined["DAX"].corr(combined["CAC40"])

    # test di Engle-Granger (statsmodels.tsa.stattools.coint fa già
    # regressione + ADF sui residui in un solo passo)
    score, pvalue, _ = coint(combined["DAX"], combined["CAC40"])

    # regressione manuale per ricavare l'hedge ratio (beta) e lo spread,
    # utile per capire la relazione anche se non serve al test in sé
    beta = np.polyfit(combined["CAC40"], combined["DAX"], 1)[0]
    spread = combined["DAX"] - beta * combined["CAC40"]
    spread_mean = spread.mean()
    spread_std = spread.std()

    cointegrata = pvalue < 0.05

    return {
        "periodo": label, "n_giorni": n, "correlazione": correlation,
        "coint_pvalue": pvalue, "cointegrata_p<0.05": cointegrata,
        "hedge_ratio_beta": beta, "spread_medio": spread_mean, "spread_std": spread_std,
    }


def main():
    print("=== Test di cointegrazione DAX-CAC40 (Engle-Granger) ===\n")
    print("Coppia effettivamente validata dallo studio citato in chat (FTSE100 già")
    print("testata e scartata il 17/07/2026, 0/5 periodi cointegrati).\n")

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
    df.to_csv("pairs_cointegration_results_cac40.csv", index=False)

    n_ok = sum(1 for r in results if r.get("cointegrata_p<0.05"))
    n_tot = sum(1 for r in results if "errore" not in r)
    print(f"=== RIEPILOGO: cointegrata in {n_ok}/{n_tot} periodi testati ===")
    if n_ok == n_tot:
        print("Cointegrazione stabile su tutti i periodi — base statistica solida per procedere.")
    elif n_ok == 0:
        print("MAI cointegrata in nessun periodo — nemmeno DAX-CAC40 è adatta al pairs trading "
              "classico su questi dati/periodo. Il filone pairs trading andrebbe chiuso.")
    else:
        print("Cointegrazione INSTABILE nel tempo — presente in alcuni periodi, assente in altri. "
              "Rischio analogo a quello già visto con ORB+ADX: nessuna base solida e stabile.")

    print("\nFile: pairs_cointegration_results_cac40.csv")


if __name__ == "__main__":
    main()
