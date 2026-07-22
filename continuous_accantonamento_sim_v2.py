"""
continuous_accantonamento_sim_v2.py — Simulazione continua 2015-2026
CORRETTA rispetto al tentativo precedente (chat 21/07/2026): qui
l'accantonamento agisce IN TEMPO REALE, mese per mese, e il capitale
ridotto alimenta davvero il sizing del mese successivo — non piu'
calcolato post-hoc su una curva libera di comporre senza freno (che
aveva prodotto una crescita esponenziale irrealistica, capitale
investito V6 a 290.000 EUR entro il 2026 — artefatto, non un risultato
reale).

METODO: il motore V6 (BacktestEngineFloatingKillSwitch) e MR
(BacktestEngineMeanReversion) vengono eseguiti UN MESE ALLA VOLTA (139
cicli), portando avanti capital_v6/capital_mr come punto di partenza
del mese successivo. A ogni fine mese si applica la STESSA formula di
consolidamento combinato di apply_monthly_consolidation_if_needed() in
live_execute.py (threshold_mult=1.5, consolidate_pct=0.4), e i pool
per il mese successivo partono gia' ridotti se scattato.

APPROSSIMAZIONE DICHIARATA: tagliare l'esecuzione a confini di mese
puo' troncare un trade aperto a cavallo tra un mese e l'altro (max
holding 48 barre = 24h, quindi al massimo ~1 giorno di distorsione ai
bordi) — effetto minore, accettato per restare fedeli alla dinamica
mensile dell'accantonamento che e' l'oggetto vero di questa analisi.

Nessuna modifica a engine.py/engine_floating_kill_switch.py/
engine_mean_reversion.py.
"""
import os
import calendar
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

CAPITAL_V6_0 = 1400.0
CAPITAL_MR_0 = 600.0
CONSOLIDATE_PCT = 0.4
THRESHOLD_MULT = 1.5
MR_MODE = "rsi"


def slice_month(signals: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    last_day = calendar.monthrange(year, month)[1]
    end = pd.Timestamp(year=year, month=month, day=last_day, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start) & (signals["timestamp"] < end)].reset_index(drop=True)


def run_month(engine_cls, capital0: float, signals_by_instrument: dict, year: int, month: int) -> float:
    """Ritorna il PnL del mese (0.0 se nessun trade)."""
    sliced = {name: slice_month(sig, year, month) for name, sig in signals_by_instrument.items()}
    if all(len(s) == 0 for s in sliced.values()):
        return 0.0
    engine_ = engine_cls(capital0=capital0)
    trades_df, _ = engine_.run(sliced)
    return float(trades_df["pnl"].sum()) if len(trades_df) else 0.0


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {}
    for name in ("DAX", "FTSE100"):
        hist[name] = get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN)
        print(f"  {name}: {len(hist[name])} barre")

    print("\nGenero segnali V6 continui...")
    v6_signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Genero segnali MR continui...")
    mr_signals = {name: generate_mean_reversion_signals(hist[name], eng.INSTRUMENTS[name], mode=MR_MODE)
                  for name in hist}

    start = hist["DAX"]["timestamp"].min()
    end = hist["DAX"]["timestamp"].max()
    months = pd.date_range(start.normalize(), end.normalize(), freq="MS", tz="UTC")

    capital_v6 = CAPITAL_V6_0
    capital_mr = CAPITAL_MR_0
    accantonato = 0.0
    reference = CAPITAL_V6_0 + CAPITAL_MR_0
    threshold = reference * THRESHOLD_MULT
    prev_year = None

    yearly_snapshot = {}
    monthly_log = []
    n_consolidations = 0

    print(f"\nEseguo {len(months)} mesi in sequenza (V6+MR, capitale reale mese-per-mese)...")

    for i, m in enumerate(months):
        year, month = m.year, m.month

        pnl_v6_month = run_month(BacktestEngineFloatingKillSwitch, capital_v6, v6_signals, year, month)
        pnl_mr_month = run_month(BacktestEngineMeanReversion, capital_mr, mr_signals, year, month)

        capital_v6 += pnl_v6_month
        capital_mr += pnl_mr_month
        combined = capital_v6 + capital_mr

        # consolidamento combinato a fine mese (stessa formula di live_execute.py)
        while combined > threshold:
            gain = combined - reference
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            reduction_fraction = consolidated / combined
            capital_v6 -= capital_v6 * reduction_fraction
            capital_mr -= capital_mr * reduction_fraction
            accantonato += consolidated
            combined = capital_v6 + capital_mr
            reference = combined
            threshold = reference * THRESHOLD_MULT
            n_consolidations += 1

        monthly_log.append({"month": m, "pnl_v6": pnl_v6_month, "pnl_mr": pnl_mr_month,
                             "cap_v6": capital_v6, "cap_mr": capital_mr,
                             "accantonato": accantonato, "combined_investito": combined,
                             "patrimonio_totale": combined + accantonato})

        if year != prev_year:
            print(f"  [{year}] investito V6={capital_v6:.2f} MR={capital_mr:.2f} "
                  f"accantonato={accantonato:.2f} patrimonio={combined+accantonato:.2f}")
            prev_year = year
        yearly_snapshot[year] = monthly_log[-1]

    print("\n=== SNAPSHOT FINE ANNO (accantonamento IN TEMPO REALE) ===")
    print(f"{'Anno':<6}{'Investito V6':>14}{'Investito MR':>14}{'Accantonato':>14}{'Patrimonio tot':>16}")
    for year in sorted(yearly_snapshot):
        s = yearly_snapshot[year]
        print(f"{year:<6}{s['cap_v6']:>14.2f}{s['cap_mr']:>14.2f}{s['accantonato']:>14.2f}"
              f"{s['patrimonio_totale']:>16.2f}")

    final = monthly_log[-1]
    print("\n=== RIEPILOGO FINALE (2026, dato parziale fino a meta' luglio) ===")
    print(f"Numero di consolidamenti avvenuti in 11 anni: {n_consolidations}")
    print(f"Accantonato finale (rendita cumulata, MAI prelevata fisicamente): {final['accantonato']:.2f} EUR")
    print(f"Capitale ancora investito (V6+MR): {final['combined_investito']:.2f} EUR")
    print(f"Patrimonio totale: {final['patrimonio_totale']:.2f} EUR (partito da "
          f"{CAPITAL_V6_0+CAPITAL_MR_0:.2f} EUR)")

    accantonato_series = [m["accantonato"] for m in monthly_log]
    max_mesi_fermi = 0
    correnti_fermi = 0
    for i in range(1, len(accantonato_series)):
        if accantonato_series[i] == accantonato_series[i - 1]:
            correnti_fermi += 1
            max_mesi_fermi = max(max_mesi_fermi, correnti_fermi)
        else:
            correnti_fermi = 0
    print(f"Mesi consecutivi piu' lunghi senza alcun consolidamento (rendita ferma): {max_mesi_fermi}")

    # capitale investito minimo/massimo osservato (per farsi un'idea della dimensione reale dei pool)
    min_combined = min(m["combined_investito"] for m in monthly_log)
    max_combined = max(m["combined_investito"] for m in monthly_log)
    print(f"Capitale investito (V6+MR) minimo/massimo osservato nei 139 mesi: "
          f"{min_combined:.2f} / {max_combined:.2f} EUR")

    print("\nNOTA: accantonamento applicato IN TEMPO REALE mese per mese, il capitale "
          "ridotto alimenta il sizing del mese successivo (a differenza del tentativo "
          "precedente). Approssimazione dichiarata: trade a cavallo tra due mesi "
          "possono essere leggermente distorti dal confine mensile (max ~1 giorno).")


if __name__ == "__main__":
    main()
