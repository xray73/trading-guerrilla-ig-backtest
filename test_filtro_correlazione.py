"""
test_filtro_correlazione.py — TEST CAUSALE COMPLETO (protocollo pieno):
filtro "salta il trade V6 quando la correlazione rolling DAX-FTSE100
(return bar-a-bar, finestra 7 giorni) e' >= 0.70", motivato dal conto
grezzo di oggi — unico segnale della sessione coerente in direzione su
5 finestre indipendenti (3/5/7/14/21 giorni) e su tutti i confronti di
quartile estremi, anche se di ampiezza modesta (0.05-0.12R/trade).

REGOLA FISSATA PRIMA DI VEDERE I RISULTATI:
- Finestra 7 giorni (delta piu' marcato nel conto grezzo: -0.121R/trade)
- Soglia 0.70 (vicina alla mediana empirica ~0.72-0.73, osservata
  stabile su tutte le finestre)
- Filtro applicato a ENTRAMBI gli strumenti (DAX e FTSE100) — la
  correlazione e' un regime di mercato condiviso, non una proprieta'
  del singolo strumento (a differenza dell'ATR%, gia' testato ieri,
  specifico di DAX)

METODO: nessuna sottoclasse di engine.py — il filtro agisce SOLO sui
segnali in input (signal='none' dove corr_rolling>=0.70), stesso
pattern del test ATR% di ieri. engine.py invariato.

Bootstrap: resampling a blocchi di giornata, N=2000, delta = PnL_filtrato
- PnL_baseline per giorno di calendario, sui 5 periodi ufficiali
combinati — stessa metodologia di tutti i test di oggi.
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
CORR_WINDOW_DAYS = 7
CORR_THRESHOLD = 0.70
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

    print(f"\nCalcolo correlazione rolling ({CORR_WINDOW_DAYS} giorni) DAX-FTSE100...")
    returns = {name: hist[name].set_index("timestamp")["close"].pct_change() for name in hist}
    aligned = pd.concat([returns["DAX"].rename("dax"), returns["FTSE100"].rename("ftse")],
                         axis=1, sort=True).dropna()
    rolling_corr = aligned["dax"].rolling(f"{CORR_WINDOW_DAYS}D").corr(aligned["ftse"]).dropna()
    corr_df = rolling_corr.rename("corr").to_frame().reset_index().rename(columns={"timestamp": "corr_time"})
    print(f"  {len(corr_df)} punti di correlazione calcolati, mediana={corr_df['corr'].median():.3f}")

    print("\nGenero segnali V6 baseline (invariati)...")
    baseline_signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print(f"Costruisco segnali FILTRATI (entrambi gli strumenti: signal='none' dove corr>={CORR_THRESHOLD})...")
    filtered_signals = {}
    for name, df in baseline_signals.items():
        df = df.copy()
        merged = pd.merge_asof(df.sort_values("timestamp"), corr_df.sort_values("corr_time"),
                                left_on="timestamp", right_on="corr_time", direction="backward")
        mask = (merged["corr"] >= CORR_THRESHOLD) & (merged["signal"].isin(["long", "short"]))
        n_masked = mask.sum()
        df.loc[mask.values, "signal"] = "none"
        filtered_signals[name] = df
        print(f"  {name}: {n_masked} segnali annullati su tutto lo storico")

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
