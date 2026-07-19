"""
analyze_vix_regime.py — Analisi ad-hoc (19/07/2026): il regime di
volatilità GENERALE del mercato (VIX, non l'ATR specifico di DAX/
FTSE100) spiega perché pattern trovati nell'esplorazione (es. filtro
ADX×ATR) funzionano in alcuni periodi e falliscono in altri?

MOTIVAZIONE: sia il filtro ADX×ATR sia (parzialmente) il test MR-solo-
FTSE100 hanno mostrato un comportamento incoerente tra i 5 periodi
ufficiali, senza che nessun parametro tecnico derivato dal prezzo di
DAX/FTSE100 (durata trend, spread EMA, distanza EMA200, ampiezza
Bollinger) riuscisse a spiegarlo. Ipotesi: la spiegazione potrebbe
stare nel contesto macro/di mercato più ampio, non nel prezzo isolato
dei due strumenti.

Dati: VIX storico giornaliero (Yahoo Finance, ticker ^VIX, via
yfinance) — indice di volatilità implicita generale del mercato
azionario USA, ampiamente usato come proxy di "paura di mercato"
complessiva, non specifico di un singolo indice.

Output: statistiche VIX medie/mediane/percentili per ciascuno dei 5
periodi ufficiali, per confronto diretto con i risultati già trovati
(es. periodi dove il blocco ADX×ATR era "buono" vs "cattivo").

Nessuna scrittura su D1. Solo stampa a log + file risultati/ per l'artifact.
"""

from __future__ import annotations

import os
import pandas as pd
import yfinance as yf

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-19"),
}

# risultati già noti dall'esplorazione del 18-19/07/2026, per confronto diretto
WIN_RATE_BLOCCO_ADX_ATR_NOTO = {
    "2015-2016": 35.9,   # sopra soglia, filtro ha danneggiato
    "2020-covid": 37.8,  # sopra soglia, filtro ha danneggiato
    "2023": None,        # campione DAX troppo piccolo, dominato da FTSE100 (49.5%)
    "2024-2025": 37.7,   # sopra soglia, ma filtro ha comunque aiutato nel backtest
    "2026-ytd": 26.4,    # sotto soglia, filtro ha aiutato
}


def main():
    os.makedirs("results", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Regime VIX vs i 5 periodi ufficiali ===\n")
    log("Scarico storico VIX (Yahoo Finance, ^VIX)...")

    vix = yf.download("^VIX", start="2014-10-01", end="2026-07-19", progress=False)
    if vix.empty:
        log("ERRORE: nessun dato VIX scaricato.")
        with open("results/analyze_vix_regime.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # yfinance con MultiIndex colonne in alcune versioni — normalizza
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"])
    log(f"  Scaricate {len(vix)} barre giornaliere, {vix['Date'].min().date()} -> {vix['Date'].max().date()}\n")

    summary_rows = []

    log(f"{'Periodo':<12} {'VIX medio':>10} {'mediana':>9} {'min':>7} {'max':>7} "
        f"{'%gg VIX>25':>11} {'%gg VIX<15':>11}")
    for label, (start, end) in PERIODS.items():
        mask = (vix["Date"] >= start) & (vix["Date"] < end)
        sub = vix.loc[mask, "Close"]
        if sub.empty:
            log(f"{label:<12} nessun dato VIX in questo range.")
            continue
        media = sub.mean()
        mediana = sub.median()
        vmin, vmax = sub.min(), sub.max()
        pct_alto = (sub > 25).mean() * 100
        pct_basso = (sub < 15).mean() * 100
        log(f"{label:<12} {media:>10.2f} {mediana:>9.2f} {vmin:>7.2f} {vmax:>7.2f} "
            f"{pct_alto:>10.1f}% {pct_basso:>10.1f}%")
        summary_rows.append({
            "periodo": label, "vix_medio": media, "vix_mediana": mediana,
            "vix_min": vmin, "vix_max": vmax,
            "pct_giorni_vix_alto": pct_alto, "pct_giorni_vix_basso": pct_basso,
            "win_rate_blocco_adx_atr_noto": WIN_RATE_BLOCCO_ADX_ATR_NOTO.get(label),
        })

    log("\n" + "=" * 70)
    log("CONFRONTO — VIX medio del periodo vs qualità nota del blocco ADX×ATR (DAX)")
    log("=" * 70)
    log(f"{'Periodo':<12} {'VIX medio':>10} {'Win rate blocco (noto)':>24} {'Interpretazione':>20}")
    for row in summary_rows:
        wr = row["win_rate_blocco_adx_atr_noto"]
        wr_str = f"{wr:.1f}%" if wr is not None else "n/d (n piccolo)"
        log(f"{row['periodo']:<12} {row['vix_medio']:>10.2f} {wr_str:>24}")

    pd.DataFrame(summary_rows).to_csv("results/vix_regime_per_periodo.csv", index=False)
    with open("results/analyze_vix_regime.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
