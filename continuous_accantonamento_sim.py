"""
continuous_accantonamento_sim.py — Simulazione CONTINUA 2015-2026 (non i
5 periodi ufficiali isolati) di V6+MR con accantonamento sul capitale
COMBINATO, per rispondere alla domanda: "che aspetto avrebbe davvero la
rendita accumulata nel tempo?"

APPROSSIMAZIONE DICHIARATA (importante, non nascosta): V6 e MR girano
qui con BacktestEngineFloatingKillSwitch/BacktestEngineMeanReversion
SENZA accantonamento innestato nel motore (capitale libero di comporre
durante il run). L'accantonamento sul capitale COMBINATO viene
applicato DOPO, ricostruendo la traiettoria capital_v6(t)+capital_mr(t)
mese per mese e applicando la STESSA formula di
apply_monthly_consolidation_if_needed() di live_execute.py. Questo
significa che il sizing dei trade durante il run NON riflette la
riduzione di capitale che l'accantonamento avrebbe causato in tempo
reale (leggermente ottimistico su quanto capitale resta investito) —
una simulazione causale vera richiederebbe un motore combinato V6+MR
con accantonamento condiviso in tempo reale, non ancora costruito.
Questo e' un primo sguardo direzionale, non un risultato validato.

Nessuna modifica a engine.py/engine_floating_kill_switch.py/
engine_mean_reversion.py — tutti importati e usati cosi' come sono.
"""
import os
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

CAPITAL_V6 = 1400.0
CAPITAL_MR = 600.0
CONSOLIDATE_PCT = 0.4
THRESHOLD_MULT = 1.5
MR_MODE = "rsi"


def build_capital_series(trades_df, capital0):
    """Da trades_df (colonne entry_time, exit_time, pnl) ricostruisce
    una serie (timestamp, capitale_dopo) — step function, un punto per
    ogni chiusura, ordinata cronologicamente per exit_time."""
    df = trades_df.sort_values("exit_time").reset_index(drop=True)
    df["capital_after"] = capital0 + df["pnl"].cumsum()
    return df[["exit_time", "capital_after"]]


def capital_at(series, ts, capital0):
    """Capitale al tempo ts: l'ultimo capital_after con exit_time<=ts,
    o capital0 se nessun trade ancora chiuso."""
    past = series[series["exit_time"] <= ts]
    if past.empty:
        return capital0
    return float(past.iloc[-1]["capital_after"])


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {}
    for name in ("DAX", "FTSE100"):
        hist[name] = get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN)
        print(f"  {name}: {len(hist[name])} barre, {hist[name]['timestamp'].min()} -> "
              f"{hist[name]['timestamp'].max()}")

    print("\nGenero segnali V6 continui...")
    v6_signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Genero segnali MR continui...")
    mr_signals = {name: generate_mean_reversion_signals(hist[name], eng.INSTRUMENTS[name], mode=MR_MODE)
                  for name in hist}

    print("\nEseguo motore V6 (BacktestEngineFloatingKillSwitch, capital0=1400) su 11 anni continui...")
    v6_engine = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    v6_trades, v6_metrics = v6_engine.run(v6_signals)
    print(f"  V6: {len(v6_trades)} trade, PnL totale {v6_trades['pnl'].sum():+.2f} EUR, "
          f"capitale finale {CAPITAL_V6 + v6_trades['pnl'].sum():.2f} EUR")

    print("\nEseguo motore MR (BacktestEngineMeanReversion, capital0=600) su 11 anni continui...")
    mr_engine = BacktestEngineMeanReversion(capital0=CAPITAL_MR)
    mr_trades, mr_metrics = mr_engine.run(mr_signals)
    print(f"  MR: {len(mr_trades)} trade, PnL totale {mr_trades['pnl'].sum():+.2f} EUR, "
          f"capitale finale {CAPITAL_MR + mr_trades['pnl'].sum():.2f} EUR")

    v6_series = build_capital_series(v6_trades, CAPITAL_V6)
    mr_series = build_capital_series(mr_trades, CAPITAL_MR)

    start = hist["DAX"]["timestamp"].min().normalize()
    end = hist["DAX"]["timestamp"].max().normalize()
    month_ends = pd.date_range(start, end, freq="MS", tz="UTC")  # inizio di ogni mese

    print(f"\nRicostruisco traiettoria mensile e applico accantonamento combinato "
          f"({len(month_ends)} mesi)...")

    accantonato = 0.0
    reference = CAPITAL_V6 + CAPITAL_MR
    threshold = reference * THRESHOLD_MULT
    prev_month = None
    yearly_snapshot = {}
    monthly_log = []
    n_consolidations = 0

    for month_start in month_ends:
        cap_v6 = capital_at(v6_series, month_start, CAPITAL_V6)
        cap_mr = capital_at(mr_series, month_start, CAPITAL_MR)
        combined = cap_v6 + cap_mr

        this_month = (month_start.year, month_start.month)
        if prev_month is not None and this_month != prev_month:
            while combined > threshold:
                gain = combined - reference
                consolidated = CONSOLIDATE_PCT * gain
                if consolidated <= 0:
                    break
                reduction_fraction = consolidated / combined
                cap_v6 -= cap_v6 * reduction_fraction
                cap_mr -= cap_mr * reduction_fraction
                accantonato += consolidated
                combined = cap_v6 + cap_mr
                reference = combined
                threshold = reference * THRESHOLD_MULT
                n_consolidations += 1
        prev_month = this_month

        monthly_log.append({"month": month_start, "cap_v6": cap_v6, "cap_mr": cap_mr,
                             "accantonato": accantonato, "combined_investito": combined,
                             "patrimonio_totale": combined + accantonato})
        yearly_snapshot[month_start.year] = monthly_log[-1]

    print("\n=== SNAPSHOT FINE ANNO (accantonato = rendita cumulata, non prelevata) ===")
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
          f"{CAPITAL_V6+CAPITAL_MR:.2f} EUR)")

    # quanti mesi consecutivi l'accantonato resta fermo (nessun consolidamento) - il "silenzio" della rendita
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

    print("\nNOTA: accantonamento calcolato POST-HOC sulla traiettoria di capitale "
          "(il sizing dei trade durante il run NON riflette la riduzione reale) — "
          "vedi docstring per il dettaglio dell'approssimazione dichiarata.")


if __name__ == "__main__":
    main()
