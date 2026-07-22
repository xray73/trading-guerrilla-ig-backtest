"""
test_regime_size_modulation.py — TEST CAUSALE COMPLETO: invece di
SALTARE il trade in regime di correlazione alta (bocciato ieri, z=-2.35
— rimuoveva trade genuinamente buoni in blocco, es. 2020-covid), qui si
DIMEZZA IL RISCHIO (size) quando corr_dax_ftse_7d >= 0.70 al momento
dell'apertura. Partecipazione ridotta ma non nulla nel regime, invece
di eliminazione totale.

Sottoclasse di BacktestEngineFloatingKillSwitch (mai modificato
engine.py/engine_floating_kill_switch.py direttamente) — override SOLO
di _position_size() (identica all'originale + moltiplicatore) e
_open_position() (per propagare il timestamp corrente alla size).

ATTENZIONE (dichiarata prima di vedere risultati): il meccanismo
"forza al minimo negoziabile" del motore originale puo' ANNULLARE
l'effetto della riduzione se la size ridotta scende sotto il minimo
IG — in quel caso torna comunque al minimo, non al 50%. Con capitale
1400 EUR e size gia' spesso vicine al minimo, questo potrebbe
attenuare molto l'effetto pratico. Lo si scopre solo testando, non
lo si assume.

Regime lookup: riusa market_regime_indicators (corr_dax_ftse_7d per
barra, gia' calcolato) invece di ricalcolare la correlazione da zero.
"""
import os
import numpy as np
import pandas as pd
import requests

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

CAPITAL_V6 = 1400.0
CORR_THRESHOLD = 0.70
SIZE_MULTIPLIER_HIGH = 0.5
N_BOOTSTRAP = 2000

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]["results"]


class BacktestEngineRegimeSized(BacktestEngineFloatingKillSwitch):
    """Come l'originale, ma la size viene calcolata su un risk_amount
    ridotto quando il regime (correlazione DAX-FTSE100 rolling 7gg) e'
    'alto' al momento dell'apertura. Nessuna altra logica toccata."""

    def __init__(self, capital0, regime_lookup: pd.Series, **kwargs):
        super().__init__(capital0, **kwargs)
        self.regime_lookup = regime_lookup  # pd.Series indicizzata per timestamp
        self._current_entry_time = None

    def _lookup_corr(self, ts):
        idx = self.regime_lookup.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return self.regime_lookup.iloc[idx]

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        self._current_entry_time = bar["timestamp"]
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)

    def _position_size(self, entry_price, stop_price, inst):
        risk_amount = self.capital * inst.risk_pct

        corr_val = self._lookup_corr(self._current_entry_time)
        if corr_val is not None and corr_val >= CORR_THRESHOLD:
            risk_amount *= SIZE_MULTIPLIER_HIGH

        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0, False, False

        size = risk_amount / (risk_distance * inst.point_value)
        forced_min_size = False
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
            forced_min_size = True

        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < size:
                size = max(max_size_by_margin, 0.0)
                margin_reduced = True

        return size, risk_amount, forced_min_size, margin_reduced


def slice_period(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def run_period_baseline(signals_by_instrument, start, end):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    trades_df, _ = engine_.run(sliced)
    return trades_df


def run_period_regime_sized(signals_by_instrument, start, end, regime_lookup):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineRegimeSized(capital0=CAPITAL_V6, regime_lookup=regime_lookup)
    trades_df, _ = engine_.run(sliced)
    return trades_df


def daily_pnl(trades_df, start, end):
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    if trades_df.empty:
        return pd.Series(0.0, index=idx)
    df = trades_df.copy()
    df["exit_day"] = pd.to_datetime(df["exit_time"]).dt.floor("D")
    daily = df.groupby("exit_day")["pnl"].sum()
    return daily.reindex(idx, fill_value=0.0)


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6 baseline...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico corr_dax_ftse_7d da market_regime_indicators (gia' calcolato)...")
    rows = d1("SELECT timestamp, corr_dax_ftse_7d FROM market_regime_indicators "
              "WHERE instrument='DAX' AND corr_dax_ftse_7d IS NOT NULL ORDER BY timestamp ASC")
    regime_lookup = pd.Series(
        [r["corr_dax_ftse_7d"] for r in rows],
        index=pd.to_datetime([r["timestamp"] for r in rows], utc=True)
    )
    print(f"  {len(regime_lookup)} punti di regime caricati")

    all_delta_days = []
    period_summary = []

    for period_name, (start, end) in PERIODS.items():
        print(f"\n=== Periodo {period_name} ===")
        baseline_trades = run_period_baseline(signals, start, end)
        regime_trades = run_period_regime_sized(signals, start, end, regime_lookup)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        regime_pnl = float(regime_trades["pnl"].sum()) if len(regime_trades) else 0.0
        delta_pnl = regime_pnl - baseline_pnl

        print(f"  Baseline: {len(baseline_trades)} trade, PnL {baseline_pnl:+.2f} EUR")
        print(f"  Size modulata: {len(regime_trades)} trade, PnL {regime_pnl:+.2f} EUR")
        print(f"  Delta: {delta_pnl:+.2f} EUR")

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_regime = daily_pnl(regime_trades, start, end)
        delta_series = (d_regime - d_baseline)
        all_delta_days.append(delta_series)

        period_summary.append({"period": period_name, "baseline_pnl": baseline_pnl,
                                "regime_pnl": regime_pnl, "delta": delta_pnl,
                                "baseline_trades": len(baseline_trades), "regime_trades": len(regime_trades)})

    combined_deltas = pd.concat(all_delta_days).values
    observed_delta = combined_deltas.sum()
    n_days_total = len(combined_deltas)

    print(f"\n=== BOOTSTRAP (blocchi di giornata, N={N_BOOTSTRAP}) ===")
    print(f"Delta osservato (reale): {observed_delta:+.2f} EUR")

    rng = np.random.default_rng(42)
    boot_sums = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(combined_deltas, size=n_days_total, replace=True)
        boot_sums[i] = sample.sum()

    std_boot = boot_sums.std()
    z_score = observed_delta / std_boot if std_boot > 0 else float("nan")
    pct_leq_zero = (boot_sums <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_sums, [2.5, 97.5])

    print(f"Z-score: {z_score:.3f}")
    print(f"%% iterazioni con delta<=0: {pct_leq_zero:.1f}%%")
    print(f"95%% CI bootstrap: [{ci_low:.2f}, {ci_high:.2f}]")

    print("\n=== RIEPILOGO PER PERIODO ===")
    print(f"{'Periodo':<12}{'Trade base':>12}{'Trade mod':>12}{'PnL base':>14}{'PnL mod':>14}{'Delta':>12}")
    for s in period_summary:
        print(f"{s['period']:<12}{s['baseline_trades']:>12}{s['regime_trades']:>12}"
              f"{s['baseline_pnl']:>14.2f}{s['regime_pnl']:>14.2f}{s['delta']:>12.2f}")

    total_baseline = sum(s["baseline_pnl"] for s in period_summary)
    total_regime = sum(s["regime_pnl"] for s in period_summary)
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)
    print(f"\nPnL totale baseline (5 periodi): {total_baseline:+.2f} EUR")
    print(f"PnL totale size-modulata (5 periodi): {total_regime:+.2f} EUR")
    print(f"Periodi migliorati: {n_periods_improved}/5")


if __name__ == "__main__":
    main()
