"""
test_filtro_atr_dax.py — TEST CAUSALE COMPLETO (protocollo pieno, non
conto grezzo): filtro "salta il trade V6 su DAX quando ATR% (atr/close
al momento del segnale) >= 0.4%", motivato dal conto grezzo di oggi
(z=-2.60 sul sottogruppo, 0.46% iterazioni casuali cosi' negative).

METODO: nessuna sottoclasse di engine.py — il filtro agisce SOLO sui
segnali in input (signal='none' dove la condizione e' vera), stesso
pattern gia' usato per gli altri filtri testati in questa sessione
(fascia oraria, ADX x volatilita', breakout eccessivo). engine.py e
BacktestEngineFloatingKillSwitch usati esattamente come sono, nessuna
modifica.

FTSE100 non toccato (il conto grezzo di oggi mostrava il pattern SOLO
su DAX, nessun pattern pulito su FTSE100).

Bootstrap: resampling a blocchi di giornata (stesso principio gia'
stabilito nel progetto, Regole_Backtest_MonteCarlo.md sez.5), N=2000,
delta = PnL_filtrato - PnL_baseline per ciascun giorno di calendario
(0 dove non cambia nulla), sui 5 periodi ufficiali combinati.
"""
import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

CAPITAL_V6 = 1400.0
ATR_PCT_THRESHOLD = 0.4  # soglia identificata nel conto grezzo di oggi
N_BOOTSTRAP = 2000

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}


def slice_period(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def run_period(signals_by_instrument: dict, start: str, end: str) -> pd.DataFrame:
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    trades_df, _metrics_df = engine_.run(sliced)
    return trades_df


def daily_pnl(trades_df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """PnL aggregato per giorno di calendario, reindicizzato sull'intero
    range del periodo (0 nei giorni senza chiusure)."""
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    if trades_df.empty:
        return pd.Series(0.0, index=idx)
    df = trades_df.copy()
    df["exit_day"] = pd.to_datetime(df["exit_time"]).dt.floor("D")
    daily = df.groupby("exit_day")["pnl"].sum()
    return daily.reindex(idx, fill_value=0.0)


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {}
    for name in ("DAX", "FTSE100"):
        hist[name] = get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN)
        print(f"  {name}: {len(hist[name])} barre")

    print("\nGenero segnali V6 baseline (invariati)...")
    baseline_signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print(f"Costruisco segnali FILTRATI (DAX: signal='none' dove ATR%>={ATR_PCT_THRESHOLD})...")
    filtered_signals = {name: df.copy() for name, df in baseline_signals.items()}
    dax = filtered_signals["DAX"]
    atr_pct = (dax["atr"] / dax["close"]) * 100
    n_masked = ((atr_pct >= ATR_PCT_THRESHOLD) & (dax["signal"].isin(["long", "short"]))).sum()
    dax.loc[atr_pct >= ATR_PCT_THRESHOLD, "signal"] = "none"
    print(f"  {n_masked} segnali DAX annullati su tutto lo storico (prima del filtro per periodo)")

    all_delta_days = []
    period_summary = []

    for period_name, (start, end) in PERIODS.items():
        print(f"\n=== Periodo {period_name} ===")
        baseline_trades = run_period(baseline_signals, start, end)
        filtered_trades = run_period(filtered_signals, start, end)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        filtered_pnl = float(filtered_trades["pnl"].sum()) if len(filtered_trades) else 0.0
        delta_pnl = filtered_pnl - baseline_pnl

        print(f"  Baseline: {len(baseline_trades)} trade, PnL {baseline_pnl:+.2f} EUR")
        print(f"  Filtrato: {len(filtered_trades)} trade, PnL {filtered_pnl:+.2f} EUR")
        print(f"  Delta: {delta_pnl:+.2f} EUR")

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_filtered = daily_pnl(filtered_trades, start, end)
        delta_series = (d_filtered - d_baseline)
        all_delta_days.append(delta_series)

        period_summary.append({"period": period_name, "baseline_pnl": baseline_pnl,
                                "filtered_pnl": filtered_pnl, "delta": delta_pnl,
                                "baseline_trades": len(baseline_trades), "filtered_trades": len(filtered_trades)})

    combined_deltas = pd.concat(all_delta_days).values
    observed_delta = combined_deltas.sum()
    n_days_total = len(combined_deltas)
    n_nonzero = int((combined_deltas != 0).sum())

    print(f"\n=== BOOTSTRAP (blocchi di giornata, N={N_BOOTSTRAP}) ===")
    print(f"Giorni totali (universo resampling, 5 periodi combinati): {n_days_total}")
    print(f"Giorni con delta!=0: {n_nonzero}")
    print(f"Delta osservato (reale, non ricampionato): {observed_delta:+.2f} EUR")

    rng = np.random.default_rng(42)
    boot_sums = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(combined_deltas, size=n_days_total, replace=True)
        boot_sums[i] = sample.sum()

    std_boot = boot_sums.std()
    z_score = observed_delta / std_boot if std_boot > 0 else float("nan")
    pct_leq_zero = (boot_sums <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_sums, [2.5, 97.5])

    print(f"Media distribuzione bootstrap: {boot_sums.mean():+.2f} EUR")
    print(f"Deviazione standard bootstrap: {std_boot:.2f} EUR")
    print(f"Z-score: {z_score:.3f}")
    print(f"%% iterazioni con delta<=0: {pct_leq_zero:.1f}%%")
    print(f"95%% CI bootstrap: [{ci_low:.2f}, {ci_high:.2f}]")

    print("\n=== RIEPILOGO PER PERIODO ===")
    print(f"{'Periodo':<12}{'Trade base':>12}{'Trade filtr':>12}{'PnL base':>14}{'PnL filtr':>14}{'Delta':>12}")
    for s in period_summary:
        print(f"{s['period']:<12}{s['baseline_trades']:>12}{s['filtered_trades']:>12}"
              f"{s['baseline_pnl']:>14.2f}{s['filtered_pnl']:>14.2f}{s['delta']:>12.2f}")

    total_baseline = sum(s["baseline_pnl"] for s in period_summary)
    total_filtered = sum(s["filtered_pnl"] for s in period_summary)
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)
    print(f"\nPnL totale baseline (5 periodi): {total_baseline:+.2f} EUR")
    print(f"PnL totale filtrato (5 periodi): {total_filtered:+.2f} EUR")
    print(f"Periodi migliorati dal filtro: {n_periods_improved}/5")


if __name__ == "__main__":
    main()
