"""
analyze_buyhold_v2_refresh.py — Rifacimento (19/07/2026) del confronto
motore V6 vs buy&hold, requisito Project Charter sez. 6. Il calcolo
precedente (RCA Addendum 13/07/2026 sez. 18) usava capitale 900EUR,
spread non ancora corretto (1.2/1.0 invece di 2.5/1.5), e quasi
certamente non la variante FloatingKillSwitch (introdotta il 15/07) ne'
il ramo mean-reversion (introdotto il 17-18/07) — nessuno dei due
esisteva ancora il 13/07. Questo script ricalcola tutto con la
configurazione di produzione ATTUALE.

Aggiornamenti rispetto al calcolo originale:
  - Capitale 2.000EUR (vincolo reale corretto il 17-18/07/2026), non 900EUR
  - Spread realistico DAX 2.5pt / FTSE100 1.5pt (campione IG raccolto il
    17/07/2026, n=2-3 per strumento — campione ancora piccolo, stesso
    limite gia' dichiarato altrove nel progetto)
  - Motore BacktestEngineFloatingKillSwitch (variante in produzione live),
    non il motore base
  - Tassi ECB/BoE storici per periodo AGGIORNATI con dati reali (media
    approssimata per periodo, non un tasso settimanale esatto IG — stessa
    natura di approssimazione del calcolo originale, ma numeri piu'
    accurati, verificati via ricerca web il 19/07/2026):
      ECB deposit facility rate medio: 2015-16 ~-0.30%, 2020 ~-0.50%,
        2023 ~3.00%, 2024-25 ~3.00%, 2026-ytd ~2.00%
      BoE base rate medio: 2015-16 ~0.45%, 2020 ~0.20%, 2023 ~4.60%,
        2024-25 ~4.60%, 2026-ytd ~3.75%
    Fee amministrativa IG stimata: +3%/anno sopra il benchmark (stessa
    stima del calcolo originale — NON confermata da fonte IG ufficiale
    per singolo strumento, resta un'approssimazione dichiarata).

Buy&hold a leva equivalente (margin_pct 5%, stessa del motore): posizione
LONG unica aperta all'inizio del periodo (size = capitale_allocato /
(prezzo*point_value*margin_pct), capitale interamente usato come
margine), MAI ribilanciata, chiusa a fine periodo. Funding overnight
calcolato notte per notte sul valore nozionale corrente (prezzo di
chiusura giornaliera), convenzione CFD standard: mercoledi' tripla
(copre il weekend), nessun addebito separato venerdi'/sabato/domenica —
approssimazione dichiarata, non la formula esatta IG per singolo giorno.

Tre scenari di allocazione capitale: solo DAX, solo FTSE100, 50/50 —
stesso schema del calcolo originale (RCA 13/07 sez. 18).

Nessuna scrittura su D1. Nessuna modifica a engine.py o a live_execute.py.
"""

from __future__ import annotations

import os
import dataclasses
from datetime import timedelta, date
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

REALISTIC_SPREAD = {"DAX": 2.5, "FTSE100": 1.5}
ADMIN_FEE_ANNUAL = 0.03  # stima IG, non confermata da fonte ufficiale per strumento

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

# Tassi medi per periodo (approssimazione dichiarata, non tasso
# settimanale esatto IG) — DAX usa ECB deposit facility rate (EUR),
# FTSE100 usa BoE base rate (GBP). Verificati via ricerca web 19/07/2026.
BENCHMARK_RATE = {
    "2015-2016": {"DAX": -0.0030, "FTSE100": 0.0045},
    "2020-covid": {"DAX": -0.0050, "FTSE100": 0.0020},
    "2023": {"DAX": 0.0300, "FTSE100": 0.0460},
    "2024-2025": {"DAX": 0.0300, "FTSE100": 0.0460},
    "2026-ytd": {"DAX": 0.0200, "FTSE100": 0.0375},
}


def fetch_bars(symbol_const, start, end) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_data(period_start: str, period_end: str) -> tuple[dict, dict]:
    """Ritorna sia i dati con segnale (per il motore V6) sia i grezzi
    (per il buy&hold) — un solo fetch riusato per entrambi."""
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    signal_data, raw_period = {}, {}
    for name, const in SYMBOLS.items():
        raw = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        inst = eng.INSTRUMENTS[name]
        full_signals = eng.generate_signals(raw, inst)
        signal_data[name] = full_signals[full_signals["timestamp"] >= p_start].reset_index(drop=True)
        raw_period[name] = raw[raw["timestamp"] >= p_start].reset_index(drop=True)
    return signal_data, raw_period


def funding_nights(entry_date: date, exit_date: date) -> int:
    """Conteggio 'notti di funding' tra entry e exit (esclusa l'ultima
    data), convenzione CFD standard: mercoledi' tripla (copre il
    weekend), nessun addebito separato venerdi'/sabato/domenica.
    Approssimazione dichiarata, non la formula esatta per singolo
    broker/giorno."""
    nights = 0
    d = entry_date
    while d < exit_date:
        wd = d.weekday()  # 0=lunedi' ... 6=domenica
        if wd == 2:              # mercoledi' -> tripla
            nights += 3
        elif wd in (4, 5, 6):    # venerdi'/sabato/domenica -> nessun addebito separato
            pass
        else:                     # lunedi'/martedi'/giovedi'
            nights += 1
        d += timedelta(days=1)
    return nights


def buyhold_leg(raw: pd.DataFrame, inst: eng.InstrumentConfig, capital_allocated: float,
                 admin_fee_annual: float, benchmark_rate_annual: float) -> dict:
    """Posizione LONG unica, size = capitale / (prezzo*point_value*margin_pct),
    aperta al primo prezzo disponibile del periodo, chiusa all'ultimo.
    Funding notte per notte sul nozionale a prezzo di chiusura giornaliera."""
    if raw.empty or len(raw) < 2:
        return {"pnl_gross": 0.0, "funding_cost": 0.0, "pnl_net": 0.0, "size": 0.0}

    spread = inst.spread_fixed
    entry_price = raw.iloc[0]["open"] + spread / 2
    exit_price = raw.iloc[-1]["close"] - spread / 2

    size = capital_allocated / (entry_price * inst.point_value * inst.margin_pct)
    pnl_gross = (exit_price - entry_price) * size

    daily = raw.set_index("timestamp")["close"].resample("1D").last().dropna()
    dates = daily.index.tz_localize(None).date
    prices = daily.values

    total_rate_annual = admin_fee_annual + benchmark_rate_annual
    daily_rate = total_rate_annual / 365.0

    funding_cost = 0.0
    for i in range(len(dates) - 1):
        n_nights = funding_nights(dates[i], dates[i + 1])
        if n_nights == 0:
            continue
        notional = prices[i] * size * inst.point_value
        funding_cost += notional * daily_rate * n_nights

    pnl_net = pnl_gross - funding_cost
    return {"pnl_gross": pnl_gross, "funding_cost": funding_cost, "pnl_net": pnl_net, "size": size}


def metrics_summary(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "pnl_total": 0.0}
    return {"n_trades": len(trades_df), "pnl_total": trades_df["pnl"].sum()}


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Rifacimento Buy&hold vs motore V6 — requisito Charter sez. 6 ===")
    log("Capitale 2.000EUR, spread realistico DAX 2.5pt/FTSE100 1.5pt, motore FloatingKillSwitch\n")

    realistic_instruments = dict(eng.INSTRUMENTS)
    for name in SYMBOLS:
        realistic_instruments[name] = dataclasses.replace(
            eng.INSTRUMENTS[name], spread_fixed=REALISTIC_SPREAD[name])

    rows = []
    for label, p_start, p_end in PERIODS:
        log(f"\n--- Periodo {label} ---")
        signal_data, raw_period = get_period_data(p_start, p_end)

        engine_v6 = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=realistic_instruments)
        trades_v6, _ = engine_v6.run(signal_data)
        m_v6 = metrics_summary(trades_v6)
        log(f"  Motore V6: n_trade={m_v6['n_trades']} PnL={m_v6['pnl_total']:+.2f} EUR")

        rates = BENCHMARK_RATE[label]
        dax_full = buyhold_leg(raw_period["DAX"], realistic_instruments["DAX"], CAPITAL0,
                                ADMIN_FEE_ANNUAL, rates["DAX"])
        ftse_full = buyhold_leg(raw_period["FTSE100"], realistic_instruments["FTSE100"], CAPITAL0,
                                 ADMIN_FEE_ANNUAL, rates["FTSE100"])
        dax_half = buyhold_leg(raw_period["DAX"], realistic_instruments["DAX"], CAPITAL0 / 2,
                                ADMIN_FEE_ANNUAL, rates["DAX"])
        ftse_half = buyhold_leg(raw_period["FTSE100"], realistic_instruments["FTSE100"], CAPITAL0 / 2,
                                 ADMIN_FEE_ANNUAL, rates["FTSE100"])
        combo_net = dax_half["pnl_net"] + ftse_half["pnl_net"]
        combo_gross = dax_half["pnl_gross"] + ftse_half["pnl_gross"]

        log(f"  Buy&hold solo DAX:     gross={dax_full['pnl_gross']:+.2f}  "
            f"funding={-dax_full['funding_cost']:+.2f}  net={dax_full['pnl_net']:+.2f}")
        log(f"  Buy&hold solo FTSE100: gross={ftse_full['pnl_gross']:+.2f}  "
            f"funding={-ftse_full['funding_cost']:+.2f}  net={ftse_full['pnl_net']:+.2f}")
        log(f"  Buy&hold 50/50:        gross={combo_gross:+.2f}  net={combo_net:+.2f}")

        rows.append({
            "periodo": label,
            "v6_pnl": m_v6["pnl_total"], "v6_n_trades": m_v6["n_trades"],
            "bh_dax_gross": dax_full["pnl_gross"], "bh_dax_net": dax_full["pnl_net"],
            "bh_ftse_gross": ftse_full["pnl_gross"], "bh_ftse_net": ftse_full["pnl_net"],
            "bh_5050_gross": combo_gross, "bh_5050_net": combo_net,
        })

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/analyze_buyhold_v2_refresh.csv", index=False)

    log(f"\n{'='*70}\nRIEPILOGO — somma 5 periodi ufficiali\n{'='*70}")
    log(f"Motore V6 (FloatingKillSwitch, spread realistico, capitale 2.000EUR): "
        f"{summary_df['v6_pnl'].sum():+.2f} EUR")
    log(f"Buy&hold solo DAX (netto funding):     {summary_df['bh_dax_net'].sum():+.2f} EUR")
    log(f"Buy&hold solo FTSE100 (netto funding): {summary_df['bh_ftse_net'].sum():+.2f} EUR")
    log(f"Buy&hold 50/50 (netto funding):        {summary_df['bh_5050_net'].sum():+.2f} EUR")
    log("\nLimiti dichiarati: tassi ECB/BoE medi per periodo (non tasso settimanale esatto IG),")
    log("fee amministrativa stimata 3%/anno (non confermata da fonte IG per singolo strumento),")
    log("convenzione funding mercoledi'-tripla (approssimazione standard CFD, non formula IG")
    log("esatta), dividendi non modellati (impatto atteso piccolo, stesso limite del calcolo")
    log("originale RCA 13/07 sez. 18).")

    with open("results/analyze_buyhold_v2_refresh.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
