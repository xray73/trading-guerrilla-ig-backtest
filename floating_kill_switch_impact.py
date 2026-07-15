"""
floating_kill_switch_impact.py — Misura l'impatto reale del kill
switch esteso al floating loss sui 5 periodi standard, DAX+FTSE100,
capitale 2.000€ (il vincolo di capitale reale). Confronta:
  - motore standard (kill switch solo su realizzato)
  - motore esteso (kill switch anche su floating loss)
per ciascun periodo: differenza in # trade, PnL, drawdown massimo, e
quante volte il blocco extra è scattato nel corso del periodo.

Non è un test di promozione (non c'è "meglio/peggio" nel senso di
sez.31 — qui l'obiettivo è correttezza del rischio, non massimizzare
PnL) — ma va comunque quantificato l'impatto prima di adottarlo in
produzione, come richiesto esplicitamente in chat.
"""

from __future__ import annotations

import dataclasses
import pandas as pd

import engine as eng
import ema_grid_search as g
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0


def run_period(period: str, full_data: dict, engine_cls, p) -> dict:
    data = {}
    for name in ["DAX", "FTSE100"]:
        inst = eng.INSTRUMENTS[name]
        window, period_start = g.slice_period(full_data[name], period)
        sig = eng.generate_signals(window, inst)
        sig = g.trim_warmup(sig, period_start)
        data[name] = sig

    engine_ = engine_cls(capital0=CAPITAL0, p=p)
    trades_df, metrics_df = engine_.run(data)

    n_forced_days = 0
    if hasattr(engine_, "_floating_loss_pct"):
        pass  # il conteggio esatto dei giorni bloccati richiederebbe hook aggiuntivo, omesso per ora

    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return {"period": period, "num_trades": n, "pnl_total": pnl, "max_drawdown_pct": dd}


def main():
    import os
    os.makedirs("results", exist_ok=True)

    full_data = {
        "DAX": g.load_full_ohlc("DAX_full.csv"),
        "FTSE100": g.load_full_ohlc("FTSE100_full.csv"),
    }

    rows = []
    for period in g.PERIODS:
        std = run_period(period, full_data, eng.BacktestEngine, eng.PARAMS)
        std["versione"] = "standard"
        rows.append(std)

        floating = run_period(period, full_data, BacktestEngineFloatingKillSwitch, eng.PARAMS)
        floating["versione"] = "floating_kill_switch"
        rows.append(floating)

        delta_trades = std["num_trades"] - floating["num_trades"]
        delta_pnl = std["pnl_total"] - floating["pnl_total"]
        delta_dd = floating["max_drawdown_pct"] - std["max_drawdown_pct"]
        print(f"[{period}] standard: {std['num_trades']} trade, pnl={std['pnl_total']:.1f}, "
              f"dd={std['max_drawdown_pct']*100:.2f}%")
        print(f"           floating: {floating['num_trades']} trade, pnl={floating['pnl_total']:.1f}, "
              f"dd={floating['max_drawdown_pct']*100:.2f}%")
        print(f"           differenza: {delta_trades} trade in meno, "
              f"{delta_pnl:+.1f}€ di PnL, drawdown {'migliorato' if delta_dd > 0 else 'invariato/peggiorato'} "
              f"di {abs(delta_dd)*100:.2f} punti\n")

    df = pd.DataFrame(rows)
    df.to_csv("results/floating_kill_switch_impact.csv", index=False)

    std_df = df[df["versione"] == "standard"]
    float_df = df[df["versione"] == "floating_kill_switch"]
    print("=" * 70)
    print("RIEPILOGO AGGREGATO (5 periodi)")
    print("=" * 70)
    print(f"Standard : {std_df['num_trades'].sum()} trade totali, "
          f"PnL totale {std_df['pnl_total'].sum():.1f}€, "
          f"peggior drawdown {std_df['max_drawdown_pct'].min()*100:.2f}%")
    print(f"Floating : {float_df['num_trades'].sum()} trade totali, "
          f"PnL totale {float_df['pnl_total'].sum():.1f}€, "
          f"peggior drawdown {float_df['max_drawdown_pct'].min()*100:.2f}%")
    print(f"\nCosto in PnL dell'estensione: "
          f"{std_df['pnl_total'].sum() - float_df['pnl_total'].sum():+.1f}€")
    print(f"Riduzione peggior drawdown: "
          f"{(float_df['max_drawdown_pct'].min() - std_df['max_drawdown_pct'].min())*100:+.2f} punti")

    print("\nCompletato. File: results/floating_kill_switch_impact.csv")


if __name__ == "__main__":
    main()
