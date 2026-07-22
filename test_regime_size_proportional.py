"""
test_regime_size_proportional.py — Come test_regime_size_modulation.py
ma con riduzione PROPORZIONALE della size invece del taglio secco al
50% — idea dell'utente (chat 22/07/2026): un taglio a scalino potrebbe
essere troppo brusco, una rampa continua potrebbe comportarsi meglio.

FORMULA DICHIARATA PRIMA DI VEDERE RISULTATI (nessun parametro scelto
guardando i dati):
  moltiplicatore = 1.0                                          se corr < 0.70
  moltiplicatore = max(0.3, 1.0 - 0.7*(corr-0.70)/0.30)          se corr >= 0.70
(rampa lineare, nessun taglio a 0.70, fino a 0.3x a corr=1.0)

Stessa sottoclasse BacktestEngineFloatingKillSwitch, stesso protocollo
(5 periodi ufficiali + bootstrap a blocchi di giornata).
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
MIN_MULTIPLIER = 0.3
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


def size_multiplier(corr_val):
    if corr_val is None or corr_val < CORR_THRESHOLD:
        return 1.0
    ramp = 1.0 - 0.7 * (corr_val - CORR_THRESHOLD) / 0.30
    return max(MIN_MULTIPLIER, ramp)


class BacktestEngineRegimeSizedProportional(BacktestEngineFloatingKillSwitch):
    def __init__(self, capital0, regime_lookup: pd.Series, **kwargs):
        super().__init__(capital0, **kwargs)
        self.regime_lookup = regime_lookup
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
        risk_amount *= size_multiplier(corr_val)

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


def slice_period(signals, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def run_period_baseline(signals_by_instrument, start, end):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    trades_df, _ = engine_.run(sliced)
    return trades_df


def run_period_proportional(signals_by_instrument, start, end, regime_lookup):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineRegimeSizedProportional(capital0=CAPITAL_V6, regime_lookup=regime_lookup)
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

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico corr_dax_ftse_7d...")
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
        prop_trades = run_period_proportional(signals, start, end, regime_lookup)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        prop_pnl = float(prop_trades["pnl"].sum()) if len(prop_trades) else 0.0
        delta_pnl = prop_pnl - baseline_pnl

        print(f"  Baseline: {len(baseline_trades)} trade, PnL {baseline_pnl:+.2f} EUR")
        print(f"  Proporzionale: {len(prop_trades)} trade, PnL {prop_pnl:+.2f} EUR")
        print(f"  Delta: {delta_pnl:+.2f} EUR")

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_prop = daily_pnl(prop_trades, start, end)
        delta_series = (d_prop - d_baseline)
        all_delta_days.append(delta_series)

        period_summary.append({"period": period_name, "baseline_pnl": baseline_pnl,
                                "prop_pnl": prop_pnl, "delta": delta_pnl,
                                "baseline_trades": len(baseline_trades), "prop_trades": len(prop_trades)})

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
    print(f"{'Periodo':<12}{'Trade base':>12}{'Trade prop':>12}{'PnL base':>14}{'PnL prop':>14}{'Delta':>12}")
    for s in period_summary:
        print(f"{s['period']:<12}{s['baseline_trades']:>12}{s['prop_trades']:>12}"
              f"{s['baseline_pnl']:>14.2f}{s['prop_pnl']:>14.2f}{s['delta']:>12.2f}")

    total_baseline = sum(s["baseline_pnl"] for s in period_summary)
    total_prop = sum(s["prop_pnl"] for s in period_summary)
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)
    print(f"\nPnL totale baseline (5 periodi): {total_baseline:+.2f} EUR")
    print(f"PnL totale proporzionale (5 periodi): {total_prop:+.2f} EUR")
    print(f"Periodi migliorati: {n_periods_improved}/5")


if __name__ == "__main__":
    main()
