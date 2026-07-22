"""
continuous_accantonamento_sim_v3.py — Come v2, ma con CORREZIONE
strutturale: consolidamento applicato PER POOL indipendentemente
(reference_v6/threshold_v6 separati da reference_mr/threshold_mr),
non piu' su equity combinata V6+MR.

MOTIVO (scoperto in chat 21/07/2026 osservando l'output di v2): con
consolidamento su equity combinata, il pool MR viene eroso
proporzionalmente ogni volta che V6 (che genera quasi tutto il
guadagno) supera la soglia — anche se MR non ha contribuito nulla al
superamento. Risultato osservato in v2: pool MR 620EUR (2015) -> 101EUR
(2026), quasi azzerato, con size sotto il minimo negoziabile per la
maggior parte dei segnali nella seconda meta' della simulazione.
Violazione del principio guida gia' stabilito nel progetto ("split
capital preferred over shared/router... pool di capitale separati
vincono sempre su pool combinato").

Ogni pool consolida SOLO il proprio guadagno sopra la propria soglia
(threshold_mult=1.5 identico, consolidate_pct=0.4 identico) — nessun
pool tocca l'altro.

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
    accantonato_v6 = 0.0
    accantonato_mr = 0.0
    reference_v6 = CAPITAL_V6_0
    threshold_v6 = reference_v6 * THRESHOLD_MULT
    reference_mr = CAPITAL_MR_0
    threshold_mr = reference_mr * THRESHOLD_MULT
    prev_year = None

    yearly_snapshot = {}
    monthly_log = []
    n_consolidations_v6 = 0
    n_consolidations_mr = 0

    print(f"\nEseguo {len(months)} mesi in sequenza (V6+MR, consolidamento PER POOL indipendente)...")

    for m in months:
        year, month = m.year, m.month

        pnl_v6_month = run_month(BacktestEngineFloatingKillSwitch, capital_v6, v6_signals, year, month)
        pnl_mr_month = run_month(BacktestEngineMeanReversion, capital_mr, mr_signals, year, month)

        capital_v6 += pnl_v6_month
        capital_mr += pnl_mr_month

        # consolidamento V6 INDIPENDENTE
        while capital_v6 > threshold_v6:
            gain = capital_v6 - reference_v6
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            capital_v6 -= consolidated
            accantonato_v6 += consolidated
            reference_v6 = capital_v6
            threshold_v6 = reference_v6 * THRESHOLD_MULT
            n_consolidations_v6 += 1

        # consolidamento MR INDIPENDENTE
        while capital_mr > threshold_mr:
            gain = capital_mr - reference_mr
            consolidated = CONSOLIDATE_PCT * gain
            if consolidated <= 0:
                break
            capital_mr -= consolidated
            accantonato_mr += consolidated
            reference_mr = capital_mr
            threshold_mr = reference_mr * THRESHOLD_MULT
            n_consolidations_mr += 1

        accantonato = accantonato_v6 + accantonato_mr
        combined = capital_v6 + capital_mr

        monthly_log.append({"month": m, "cap_v6": capital_v6, "cap_mr": capital_mr,
                             "accantonato_v6": accantonato_v6, "accantonato_mr": accantonato_mr,
                             "accantonato": accantonato, "combined_investito": combined,
                             "patrimonio_totale": combined + accantonato})

        if year != prev_year:
            print(f"  [{year}] investito V6={capital_v6:.2f} MR={capital_mr:.2f} "
                  f"accantonato_v6={accantonato_v6:.2f} accantonato_mr={accantonato_mr:.2f} "
                  f"patrimonio={combined+accantonato:.2f}")
            prev_year = year
        yearly_snapshot[year] = monthly_log[-1]

    print("\n=== SNAPSHOT FINE ANNO (consolidamento PER POOL) ===")
    print(f"{'Anno':<6}{'Investito V6':>13}{'Investito MR':>13}{'Accant.V6':>12}{'Accant.MR':>12}{'Patrim.tot':>14}")
    for year in sorted(yearly_snapshot):
        s = yearly_snapshot[year]
        print(f"{year:<6}{s['cap_v6']:>13.2f}{s['cap_mr']:>13.2f}{s['accantonato_v6']:>12.2f}"
              f"{s['accantonato_mr']:>12.2f}{s['patrimonio_totale']:>14.2f}")

    final = monthly_log[-1]
    print("\n=== RIEPILOGO FINALE (2026, dato parziale fino a meta' luglio) ===")
    print(f"Consolidamenti V6: {n_consolidations_v6}, consolidamenti MR: {n_consolidations_mr}")
    print(f"Accantonato V6: {final['accantonato_v6']:.2f} EUR")
    print(f"Accantonato MR: {final['accantonato_mr']:.2f} EUR")
    print(f"Accantonato totale: {final['accantonato']:.2f} EUR")
    print(f"Capitale ancora investito V6: {final['cap_v6']:.2f} EUR")
    print(f"Capitale ancora investito MR: {final['cap_mr']:.2f} EUR")
    print(f"Patrimonio totale: {final['patrimonio_totale']:.2f} EUR (partito da "
          f"{CAPITAL_V6_0+CAPITAL_MR_0:.2f} EUR)")

    print("\nNOTA: confrontare cap_mr finale qui con i 101.42 EUR ottenuti nella versione "
          "a equity combinata (v2) — se il pool MR resta vicino al suo valore iniziale "
          "(600 EUR) invece di prosciugarsi, la correzione per-pool risolve il problema "
          "strutturale osservato.")


if __name__ == "__main__":
    main()
