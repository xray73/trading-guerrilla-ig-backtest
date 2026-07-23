"""
test_ftse_continuous_multiplier_holdout.py — Validazione holdout vera
per il moltiplicatore continuo (z=2,125 su tutta la storia, ma percentili
fit sull'intero campione — stesso limite già superato per corr_ftse_gold
con train/holdout). Stesso principio: a(t)/c(t) (rango percentile di
ATR% e corr_dax_ftse_7d) calcolati SOLO sul train (tutti i periodi
tranne 2020-covid), poi applicati COSÌ COME SONO all'holdout
(2020-covid), senza ricalibrare — se il vantaggio regge su un campione
mai usato per definire la funzione, è un caso diverso dai 6 falliti.

Meccanica del fit train-only: percentile di un valore holdout = rango
che occuperebbe SE inserito nella distribuzione train (via ricerca
binaria sui valori train ordinati) — non richiede che il valore
holdout sia mai stato visto nel train, gestisce naturalmente valori
fuori range (clip a 0 o 1).

Report separato per: (a) SOLO 2020-covid (il vero test), (b) tutti e 5
i periodi con soglie fit-su-train (per confronto diretto con la
versione fit-su-tutto, z=2,125).

Stesso sanity check obbligatorio, stesso bootstrap a blocchi di
giornata, N=2000.
"""
import os
import sys
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
N_BOOTSTRAP = 2000
TARGET_INSTRUMENT = "FTSE100"
HOLDOUT_LABEL = "2020-covid"

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


def train_only_percentile(train_values: np.ndarray, all_values: pd.Series) -> pd.Series:
    """Rango percentile di ogni valore in all_values RISPETTO alla
    distribuzione train_values (ordinata) — non rispetto a se stesso.
    Valori fuori dal range train vengono clippati a 0.0/1.0."""
    sorted_train = np.sort(train_values)
    n_train = len(sorted_train)
    ranks = np.searchsorted(sorted_train, all_values.values, side="right")
    pct = ranks / n_train
    return pd.Series(np.clip(pct, 0.0, 1.0), index=all_values.index)


class BacktestEngineFtseContinuousMultiplier(BacktestEngineFloatingKillSwitch):
    def __init__(self, capital0, multiplier_lookup: pd.Series, force_neutral: bool = False, **kwargs):
        super().__init__(capital0, **kwargs)
        self.multiplier_lookup = multiplier_lookup
        self.force_neutral = force_neutral
        self._current_instrument = None
        self._current_entry_time = None

    def _lookup_multiplier(self, ts) -> float:
        if self.multiplier_lookup.empty:
            return 1.0
        idx = self.multiplier_lookup.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return 1.0
        val = self.multiplier_lookup.iloc[idx]
        return 1.0 if pd.isna(val) else float(val)

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        self._current_instrument = instrument
        self._current_entry_time = bar["timestamp"]
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)

    def _position_size(self, entry_price, stop_price, inst):
        risk_amount = self.capital * inst.risk_pct
        multiplier = 1.0
        if not self.force_neutral and self._current_instrument == TARGET_INSTRUMENT:
            multiplier = self._lookup_multiplier(self._current_entry_time)
        risk_amount *= multiplier

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


def run_period_continuous(signals_by_instrument, start, end, multiplier_lookup, force_neutral=False):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFtseContinuousMultiplier(
        capital0=CAPITAL_V6, multiplier_lookup=multiplier_lookup, force_neutral=force_neutral)
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


def bootstrap_periods(signals, multiplier_lookup, period_labels):
    all_delta_days = []
    period_summary = []
    for period_name in period_labels:
        start, end = PERIODS[period_name]
        baseline_trades = run_period_baseline(signals, start, end)
        cont_trades = run_period_continuous(signals, start, end, multiplier_lookup)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        cont_pnl = float(cont_trades["pnl"].sum()) if len(cont_trades) else 0.0

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_cont = daily_pnl(cont_trades, start, end)
        all_delta_days.append(d_cont - d_baseline)

        period_summary.append({
            "period": period_name, "baseline_pnl": baseline_pnl, "cont_pnl": cont_pnl,
            "delta": cont_pnl - baseline_pnl,
            "baseline_trades": len(baseline_trades), "cont_trades": len(cont_trades),
        })

    combined_deltas = pd.concat(all_delta_days).values
    observed_delta = combined_deltas.sum()
    n_days_total = len(combined_deltas)

    rng = np.random.default_rng(42)
    boot_sums = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(combined_deltas, size=n_days_total, replace=True)
        boot_sums[i] = sample.sum()

    std_boot = boot_sums.std()
    z_score = observed_delta / std_boot if std_boot > 0 else float("nan")
    pct_leq_zero = (boot_sums <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_sums, [2.5, 97.5])

    return {
        "observed_delta": observed_delta, "z_score": z_score, "pct_leq_zero": pct_leq_zero,
        "ci_low": ci_low, "ci_high": ci_high, "period_summary": period_summary,
    }


def print_result(label, res):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for s in res["period_summary"]:
        print(f"  {s['period']:<12} trade base={s['baseline_trades']:>4} cont={s['cont_trades']:>4}  "
              f"PnL base={s['baseline_pnl']:>10.2f}  PnL cont={s['cont_pnl']:>10.2f}  "
              f"delta={s['delta']:>+9.2f}")
    print(f"\n  Delta osservato: {res['observed_delta']:+.2f} EUR")
    print(f"  Z-score: {res['z_score']:.3f}")
    print(f"  %% iterazioni con delta<=0: {res['pct_leq_zero']:.1f}%%")
    print(f"  95%% CI bootstrap: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")


def sanity_check(signals, multiplier_lookup):
    print("=== SANITY CHECK (obbligatorio) ===")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_continuous(signals, start, end, multiplier_lookup, force_neutral=True)
    n_base, n_neutral = len(baseline), len(neutral)
    pnl_base = float(baseline["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral["pnl"].sum()) if n_neutral else 0.0
    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral: {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")
    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("\n  *** SANITY CHECK FALLITO *** — INTERROMPO.")
        sys.exit(1)
    print("  OK\n")


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico atr_pct (FTSE100) e corr_dax_ftse_7d...")
    rows_atr = d1(f"SELECT timestamp, atr_pct FROM market_regime_indicators "
                  f"WHERE instrument='{TARGET_INSTRUMENT}' AND atr_pct IS NOT NULL ORDER BY timestamp ASC")
    rows_corr = d1("SELECT timestamp, corr_dax_ftse_7d FROM market_regime_indicators "
                   "WHERE instrument='DAX' AND corr_dax_ftse_7d IS NOT NULL ORDER BY timestamp ASC")

    atr_series = pd.Series(
        [r["atr_pct"] for r in rows_atr],
        index=pd.to_datetime([r["timestamp"] for r in rows_atr], utc=True))
    corr_series = pd.Series(
        [r["corr_dax_ftse_7d"] for r in rows_corr],
        index=pd.to_datetime([r["timestamp"] for r in rows_corr], utc=True))

    combined = pd.concat([atr_series.rename("atr"), corr_series.rename("corr")], axis=1, sort=True).dropna()

    holdout_start, holdout_end = PERIODS[HOLDOUT_LABEL]
    holdout_mask = ((combined.index >= pd.Timestamp(holdout_start, tz="UTC")) &
                     (combined.index < pd.Timestamp(holdout_end, tz="UTC") + pd.Timedelta(days=1)))
    train_mask = ~holdout_mask

    print(f"  Train: {train_mask.sum()} punti (esclude {HOLDOUT_LABEL})  "
          f"Holdout: {holdout_mask.sum()} punti (solo {HOLDOUT_LABEL})")

    # --- Percentili fit SOLO su train, applicati a TUTTA la serie (train+holdout) ---
    a_holdout_fit = train_only_percentile(combined.loc[train_mask, "atr"].values, combined["atr"])
    c_holdout_fit = train_only_percentile(combined.loc[train_mask, "corr"].values, combined["corr"])
    multiplier_holdout_fit = 1.0 + (a_holdout_fit * c_holdout_fit)

    print(f"  Moltiplicatore (fit train-only) su holdout {HOLDOUT_LABEL}: "
          f"min={multiplier_holdout_fit[holdout_mask].min():.3f} "
          f"max={multiplier_holdout_fit[holdout_mask].max():.3f} "
          f"media={multiplier_holdout_fit[holdout_mask].mean():.3f}\n")

    sanity_check(signals, multiplier_holdout_fit)

    # --- Test A: SOLO il periodo holdout (2020-covid) — il test vero ---
    res_holdout_only = bootstrap_periods(signals, multiplier_holdout_fit, [HOLDOUT_LABEL])
    print_result(f"TEST A — SOLO HOLDOUT ({HOLDOUT_LABEL}), soglie fit su train", res_holdout_only)

    # --- Test B: tutti e 5 i periodi, ma con soglie fit-su-train (non su tutto) ---
    res_all_trainfit = bootstrap_periods(signals, multiplier_holdout_fit, list(PERIODS.keys()))
    print_result("TEST B — TUTTI I 5 PERIODI, soglie fit-su-train (per confronto con fit-su-tutto, z=2.125)",
                  res_all_trainfit)

    print("\n" + "=" * 70)
    print("VERDETTO FINALE")
    print("=" * 70)
    print(f"Test A (solo holdout {HOLDOUT_LABEL}): z={res_holdout_only['z_score']:.3f}, "
          f"delta={res_holdout_only['observed_delta']:+.2f} EUR")
    print(f"Test B (5 periodi, fit-su-train): z={res_all_trainfit['z_score']:.3f}, "
          f"delta={res_all_trainfit['observed_delta']:+.2f} EUR")
    print("Per riferimento, versione originale (fit-su-tutto, stesso campione del test): z=2.125")
    if res_holdout_only["z_score"] > 0 and res_holdout_only["observed_delta"] > 0:
        print("\nIl segnale regge (direzione positiva) anche su un holdout mai usato per calibrare "
              "i percentili — coerente con un effetto reale, non overfitting sull'intero campione.")
    else:
        print("\nIl segnale si indebolisce o si capovolge sull'holdout indipendente — coerente con "
              "sovradattamento all'intero campione usato per il fit originale.")


if __name__ == "__main__":
    main()
