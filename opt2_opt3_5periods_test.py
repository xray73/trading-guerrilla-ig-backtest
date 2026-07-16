"""
opt2_opt3_5periods_test.py — Scenario "accantonamento" (opzione 2:
giveback 30% su nuovo massimo storico; opzione 3: gradini +50%,
consolida 40% — sez. chat 16/07/2026) applicato ai 5 periodi storici
ufficiali del progetto (2015-2016, 2020-covid, 2023, 2024-2025,
2026-ytd), ciascuna in due varianti di frequenza del check:

  A) CHECK CONTINUO: la soglia +50% viene controllata dopo ogni singolo
     trade chiuso (come nello scenario già fatto sui 180 giorni recenti).
  B) CHECK MENSILE: la soglia viene controllata solo quando si passa a
     un nuovo mese di calendario, usando il capitale investito così
     com'era alla fine del mese precedente — anche se durante il mese
     l'equity avesse superato la soglia più volte, si consolida una
     volta sola al cambio di mese.

Ogni periodo è simulato in modo INDIPENDENTE (capitale riparte da
CAPITAL0 ogni volta) — coerente con la metodologia walk-forward del
progetto (i 5 periodi non sono contigui, ci sono buchi 2017-2019 e
2021-2022 mai testati).

Nessuna modifica al motore standard. Solo lettura per il sizing dei
trade reali (BacktestEngineFloatingKillSwitch, invariato), poi replay
con size ricalcolato — stessa tecnica di approssimazione lineare già
usata e verificata (0 trade a size forzata minima nel campione 180gg).

CONTESTO invariato: analisi descrittiva per capire il compromesso
accantonamento/crescita persa, non un test di validazione del segnale.
Nessuna scrittura su D1.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
RISK_PCT = {"DAX": 0.02, "FTSE100": 0.015}
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

# date esatte recuperate da D1 (run 7-11), fine periodo +1 giorno per includere l'ultimo trade
PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

CONSOLIDATE_PCT = 0.4   # quota consolidata di ogni gradino +50% (opzione 3)
THRESHOLD_MULT = 1.5    # +50% dall'ultimo riferimento (opzione 3)
GIVEBACK_PCT = 0.3      # quota accantonata di ogni nuovo massimo storico (opzione 2)


def fetch_bars(symbol_const, start: datetime, end: datetime, interval) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, interval, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_trades(period_start: str, period_end: str) -> pd.DataFrame:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    full_data_30m = {}
    for name, const in SYMBOLS.items():
        full_data_30m[name] = fetch_bars(const, warmup_start.to_pydatetime(),
                                          p_end.to_pydatetime(), dukascopy_python.INTERVAL_MIN_30)

    signal_data = {}
    for name in SYMBOLS:
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(full_data_30m[name], inst)

    engine_baseline = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_all, _ = engine_baseline.run(signal_data)
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"], utc=True)

    in_period = (trades_all["entry_time"] >= p_start) & (trades_all["entry_time"] < p_end)
    trades_period = trades_all[in_period].sort_values("entry_time").reset_index(drop=True)
    return trades_period


def simulate_opt2(trades: pd.DataFrame, monthly_only: bool) -> dict:
    invested = CAPITAL0
    peak = CAPITAL0
    side_pool = 0.0
    last_month = None
    equity_rows = []  # (timestamp, totale=investito+accantonato) per il drawdown

    def try_ratchet():
        nonlocal invested, peak, side_pool
        if invested > peak:
            increment = invested - peak
            skim = GIVEBACK_PCT * increment
            side_pool += skim
            invested -= skim
            peak = invested + skim  # nuovo massimo storico raggiunto (pre-accantonamento)

    for _, t in trades.iterrows():
        trade_month = (t["entry_time"].year, t["entry_time"].month)

        if monthly_only:
            if last_month is not None and trade_month != last_month:
                try_ratchet()  # check solo al cambio di mese

        instr = t["instrument"]
        risk_amount = RISK_PCT[instr] * invested
        orig_risk = t["risk_amount"]
        scale = risk_amount / orig_risk if orig_risk != 0 else 1.0
        scaled_pnl = t["pnl"] * scale
        invested += scaled_pnl

        if not monthly_only:
            try_ratchet()  # check continuo, dopo ogni trade

        last_month = trade_month
        equity_rows.append((t["exit_time"], invested + side_pool))

    if monthly_only:
        try_ratchet()  # check finale di chiusura periodo

    max_dd_pct, max_dd_eur = compute_drawdown(equity_rows, CAPITAL0)

    return {
        "n_trades": len(trades),
        "invested_finale": invested,
        "accantonato_finale": side_pool,
        "totale_finale": invested + side_pool,
        "rendimento_pct": 100 * (invested + side_pool - CAPITAL0) / CAPITAL0,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_eur": max_dd_eur,
    }


def compute_drawdown(equity_rows: list[tuple], capital0: float) -> tuple[float, float]:
    """equity_rows: [(timestamp, totale), ...] in ordine cronologico.
    Ritorna (max_drawdown_pct, max_drawdown_eur)."""
    if not equity_rows:
        return 0.0, 0.0
    equity = pd.Series([r[1] for r in equity_rows])
    running_max = equity.cummax()
    dd_eur = equity - running_max
    dd_pct = dd_eur / running_max
    return dd_pct.min() * 100, dd_eur.min()


def simulate_opt3(trades: pd.DataFrame, monthly_only: bool) -> dict:
    invested = CAPITAL0
    reference = CAPITAL0
    threshold = reference * THRESHOLD_MULT
    side_pool = 0.0
    last_month = None
    equity_rows = []

    def try_consolidate():
        nonlocal invested, reference, threshold, side_pool
        while invested > threshold:
            gain = invested - reference
            consolidated = CONSOLIDATE_PCT * gain
            side_pool += consolidated
            invested -= consolidated
            reference = invested
            threshold = reference * THRESHOLD_MULT

    for _, t in trades.iterrows():
        trade_month = (t["entry_time"].year, t["entry_time"].month)

        if monthly_only:
            if last_month is not None and trade_month != last_month:
                try_consolidate()  # check solo al cambio di mese, su capitale di fine mese precedente
        # (variante continua: il check avviene dopo OGNI trade, più sotto)

        instr = t["instrument"]
        risk_amount = RISK_PCT[instr] * invested
        orig_risk = t["risk_amount"]
        scale = risk_amount / orig_risk if orig_risk != 0 else 1.0
        scaled_pnl = t["pnl"] * scale
        invested += scaled_pnl

        if not monthly_only:
            try_consolidate()  # check continuo, dopo ogni trade

        last_month = trade_month
        equity_rows.append((t["exit_time"], invested + side_pool))

    if monthly_only:
        try_consolidate()  # check finale di chiusura periodo

    max_dd_pct, max_dd_eur = compute_drawdown(equity_rows, CAPITAL0)

    final_invested = invested
    final_side_pool = side_pool
    return {
        "n_trades": len(trades),
        "invested_finale": final_invested,
        "accantonato_finale": final_side_pool,
        "totale_finale": final_invested + final_side_pool,
        "rendimento_pct": 100 * (final_invested + final_side_pool - CAPITAL0) / CAPITAL0,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_eur": max_dd_eur,
    }


def main():
    print(f"Capitale iniziale per periodo: {CAPITAL0:.0f} EUR (indipendente per ciascuno dei 5 periodi)\n")

    all_rows = []
    for label, p_start, p_end in PERIODS:
        print(f"--- Periodo {label} ({p_start} -> {p_end}) ---")
        trades = get_period_trades(p_start, p_end)
        print(f"  {len(trades)} trade nel periodo")

        if len(trades) == 0:
            print("  Nessun trade, salto.")
            continue

        res_continuo = simulate_opt3(trades, monthly_only=False)
        res_mensile = simulate_opt3(trades, monthly_only=True)
        res_opt2_continuo = simulate_opt2(trades, monthly_only=False)
        res_opt2_mensile = simulate_opt2(trades, monthly_only=True)

        # replay "reale" (nessun tetto) per confronto, stessa tecnica di scaling
        invested_real = CAPITAL0
        real_equity_rows = []
        for _, t in trades.iterrows():
            instr = t["instrument"]
            risk_amount = RISK_PCT[instr] * invested_real
            scale = risk_amount / t["risk_amount"] if t["risk_amount"] != 0 else 1.0
            invested_real += t["pnl"] * scale
            real_equity_rows.append((t["exit_time"], invested_real))
        real_dd_pct, real_dd_eur = compute_drawdown(real_equity_rows, CAPITAL0)

        print(f"  Reale (senza meccanismo):  {invested_real:>10.0f} EUR ({100*(invested_real-CAPITAL0)/CAPITAL0:+.1f}%)  "
              f"MaxDD={real_dd_pct:.1f}% ({real_dd_eur:.0f} EUR)")
        print(f"  Opt.2 check continuo:      investito={res_opt2_continuo['invested_finale']:.0f}  "
              f"accantonato={res_opt2_continuo['accantonato_finale']:.0f}  totale={res_opt2_continuo['totale_finale']:.0f} "
              f"({res_opt2_continuo['rendimento_pct']:+.1f}%)  MaxDD={res_opt2_continuo['max_drawdown_pct']:.1f}% "
              f"({res_opt2_continuo['max_drawdown_eur']:.0f} EUR)")
        print(f"  Opt.2 check mensile:       investito={res_opt2_mensile['invested_finale']:.0f}  "
              f"accantonato={res_opt2_mensile['accantonato_finale']:.0f}  totale={res_opt2_mensile['totale_finale']:.0f} "
              f"({res_opt2_mensile['rendimento_pct']:+.1f}%)  MaxDD={res_opt2_mensile['max_drawdown_pct']:.1f}% "
              f"({res_opt2_mensile['max_drawdown_eur']:.0f} EUR)")
        print(f"  Opt.3 check continuo:      investito={res_continuo['invested_finale']:.0f}  "
              f"accantonato={res_continuo['accantonato_finale']:.0f}  totale={res_continuo['totale_finale']:.0f} "
              f"({res_continuo['rendimento_pct']:+.1f}%)  MaxDD={res_continuo['max_drawdown_pct']:.1f}% "
              f"({res_continuo['max_drawdown_eur']:.0f} EUR)")
        print(f"  Opt.3 check mensile:       investito={res_mensile['invested_finale']:.0f}  "
              f"accantonato={res_mensile['accantonato_finale']:.0f}  totale={res_mensile['totale_finale']:.0f} "
              f"({res_mensile['rendimento_pct']:+.1f}%)  MaxDD={res_mensile['max_drawdown_pct']:.1f}% "
              f"({res_mensile['max_drawdown_eur']:.0f} EUR)")
        print()

        all_rows.append({
            "periodo": label, "n_trade": len(trades),
            "reale_finale": invested_real,
            "reale_rendimento_pct": 100*(invested_real-CAPITAL0)/CAPITAL0,
            "reale_max_dd_pct": real_dd_pct,
            "reale_max_dd_eur": real_dd_eur,
            "opt2_continuo_investito": res_opt2_continuo["invested_finale"],
            "opt2_continuo_accantonato": res_opt2_continuo["accantonato_finale"],
            "opt2_continuo_totale": res_opt2_continuo["totale_finale"],
            "opt2_continuo_rendimento_pct": res_opt2_continuo["rendimento_pct"],
            "opt2_continuo_max_dd_pct": res_opt2_continuo["max_drawdown_pct"],
            "opt2_continuo_max_dd_eur": res_opt2_continuo["max_drawdown_eur"],
            "opt2_mensile_investito": res_opt2_mensile["invested_finale"],
            "opt2_mensile_accantonato": res_opt2_mensile["accantonato_finale"],
            "opt2_mensile_totale": res_opt2_mensile["totale_finale"],
            "opt2_mensile_rendimento_pct": res_opt2_mensile["rendimento_pct"],
            "opt2_mensile_max_dd_pct": res_opt2_mensile["max_drawdown_pct"],
            "opt2_mensile_max_dd_eur": res_opt2_mensile["max_drawdown_eur"],
            "opt3_continuo_investito": res_continuo["invested_finale"],
            "opt3_continuo_accantonato": res_continuo["accantonato_finale"],
            "opt3_continuo_totale": res_continuo["totale_finale"],
            "opt3_continuo_rendimento_pct": res_continuo["rendimento_pct"],
            "opt3_continuo_max_dd_pct": res_continuo["max_drawdown_pct"],
            "opt3_continuo_max_dd_eur": res_continuo["max_drawdown_eur"],
            "opt3_mensile_investito": res_mensile["invested_finale"],
            "opt3_mensile_accantonato": res_mensile["accantonato_finale"],
            "opt3_mensile_totale": res_mensile["totale_finale"],
            "opt3_mensile_rendimento_pct": res_mensile["rendimento_pct"],
            "opt3_mensile_max_dd_pct": res_mensile["max_drawdown_pct"],
            "opt3_mensile_max_dd_eur": res_mensile["max_drawdown_eur"],
        })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv("opt2_opt3_5periods_summary.csv", index=False)
    print("Completato. File: opt2_opt3_5periods_summary.csv")


if __name__ == "__main__":
    main()
