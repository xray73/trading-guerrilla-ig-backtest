"""
test_ftse_composite_boost_sweep.py — Estensione di
test_ftse_composite_size_boost.py: invece di un solo moltiplicatore
(1.3x), testa un set di valori DICHIARATI ORA, prima di vedere
qualunque risultato, per verificare la robustezza del pattern rispetto
al parametro (Principio 3 del Protocollo Anti-Rumore: "un pattern che
regge solo a un valore specifico e sparisce ad altri è probabilmente
rumore centrato per caso").

MOLTIPLICATORI TESTATI (fissati ora): 1.1, 1.3, 1.5, 1.7
Nessuno di questi è scelto per "far passare" il test — sono un range
attorno al valore originale (1.3, già testato con z=1.67) per vedere
la FORMA della curva z(moltiplicatore), non il valore massimo.

INTERPRETAZIONE (dichiarata ora, prima di vedere risultati):
  - z relativamente stabile/coerente su tutto il range -> rafforza la
    fiducia nel pattern sottostante, indipendentemente da quale
    supera la soglia 2.0 (o se nessuno la supera)
  - z che sale bruscamente solo a un moltiplicatore isolato, senza una
    progressione graduale -> probabile rumore centrato per caso
    (pochi trade/giorni anomali pesano sempre di più), si considera il
    filone chiuso

Stesso identico motore/sanity-check/bootstrap di
test_ftse_composite_size_boost.py per ciascun moltiplicatore — unica
differenza è il valore di BOOST_MULTIPLIER, iterato in un ciclo.
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

# --- Criteri fissati PRIMA di vedere risultati ---
MULTIPLIERS_TO_TEST = [1.1, 1.3, 1.5, 1.7]
ATR_THRESHOLD_FTSE = 0.2031323100223204
CORR_THRESHOLD = 0.7853464827260775
TARGET_INSTRUMENT = "FTSE100"

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


class BacktestEngineFtseCompositeBoost(BacktestEngineFloatingKillSwitch):
    """Identica a test_ftse_composite_size_boost.py, con BOOST_MULTIPLIER
    passato a init invece che costante globale — necessario per iterare
    su più valori nello stesso script senza duplicare la classe."""

    def __init__(self, capital0, stress_lookup: pd.Series, boost_multiplier: float,
                 force_neutral: bool = False, **kwargs):
        super().__init__(capital0, **kwargs)
        self.stress_lookup = stress_lookup
        self.boost_multiplier = boost_multiplier
        self.force_neutral = force_neutral
        self._current_instrument = None
        self._current_entry_time = None

    def _is_stress(self, ts) -> bool:
        if self.stress_lookup.empty:
            return False
        idx = self.stress_lookup.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return False
        return bool(self.stress_lookup.iloc[idx])

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        self._current_instrument = instrument
        self._current_entry_time = bar["timestamp"]
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)

    def _position_size(self, entry_price, stop_price, inst):
        risk_amount = self.capital * inst.risk_pct

        multiplier = 1.0
        if not self.force_neutral and self._current_instrument == TARGET_INSTRUMENT:
            if self._is_stress(self._current_entry_time):
                multiplier = self.boost_multiplier
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


def run_period_boost(signals_by_instrument, start, end, stress_lookup, boost_multiplier, force_neutral=False):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFtseCompositeBoost(
        capital0=CAPITAL_V6, stress_lookup=stress_lookup,
        boost_multiplier=boost_multiplier, force_neutral=force_neutral)
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


def sanity_check(signals, stress_lookup):
    print("=== SANITY CHECK (obbligatorio, PRIMA di qualunque moltiplicatore) ===")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_boost(signals, start, end, stress_lookup, boost_multiplier=1.0, force_neutral=True)

    n_base, n_neutral = len(baseline), len(neutral)
    pnl_base = float(baseline["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral["pnl"].sum()) if n_neutral else 0.0

    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral (force_neutral=True): {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")

    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("\n  *** SANITY CHECK FALLITO *** — INTERROMPO.")
        sys.exit(1)
    print("  Sanity check OK — procedo con lo sweep.\n")


def run_bootstrap_for_multiplier(signals, stress_lookup, boost_multiplier):
    all_delta_days = []
    period_summary = []

    for period_name, (start, end) in PERIODS.items():
        baseline_trades = run_period_baseline(signals, start, end)
        boost_trades = run_period_boost(signals, start, end, stress_lookup, boost_multiplier)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        boost_pnl = float(boost_trades["pnl"].sum()) if len(boost_trades) else 0.0

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_boost = daily_pnl(boost_trades, start, end)
        all_delta_days.append(d_boost - d_baseline)

        period_summary.append({
            "period": period_name, "baseline_pnl": baseline_pnl, "boost_pnl": boost_pnl,
            "delta": boost_pnl - baseline_pnl,
        })

    combined_deltas = pd.concat(all_delta_days).values
    observed_delta = combined_deltas.sum()
    n_days_total = len(combined_deltas)

    rng = np.random.default_rng(42)  # stesso seed per ogni moltiplicatore, comparabilità diretta
    boot_sums = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(combined_deltas, size=n_days_total, replace=True)
        boot_sums[i] = sample.sum()

    std_boot = boot_sums.std()
    z_score = observed_delta / std_boot if std_boot > 0 else float("nan")
    pct_leq_zero = (boot_sums <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_sums, [2.5, 97.5])
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)

    return {
        "multiplier": boost_multiplier, "observed_delta": observed_delta, "z_score": z_score,
        "pct_leq_zero": pct_leq_zero, "ci_low": ci_low, "ci_high": ci_high,
        "n_periods_improved": n_periods_improved, "period_summary": period_summary,
    }


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico atr_pct (FTSE100) e corr_dax_ftse_7d per costruire lo stato composito...")
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
    stress_lookup = (combined["atr"] > ATR_THRESHOLD_FTSE) & (combined["corr"] > CORR_THRESHOLD)
    print(f"  {len(stress_lookup)} punti di stato, {stress_lookup.sum()} in stress "
          f"({100*stress_lookup.mean():.1f}%)\n")

    sanity_check(signals, stress_lookup)

    results = []
    for m in MULTIPLIERS_TO_TEST:
        print(f"=== Moltiplicatore {m}x ===")
        r = run_bootstrap_for_multiplier(signals, stress_lookup, m)
        results.append(r)
        print(f"  Delta osservato: {r['observed_delta']:+.2f} EUR | z={r['z_score']:.3f} | "
              f"%%<=0={r['pct_leq_zero']:.1f}%% | CI95=[{r['ci_low']:.2f}, {r['ci_high']:.2f}] | "
              f"periodi migliorati={r['n_periods_improved']}/5\n")

    print("=" * 78)
    print("RIEPILOGO SWEEP — forma della curva z(moltiplicatore)")
    print("=" * 78)
    print(f"{'Moltiplicatore':<16}{'Delta EUR':>14}{'z-score':>10}{'%%<=0':>10}{'Periodi mig.':>14}")
    for r in results:
        print(f"{r['multiplier']:<16}{r['observed_delta']:>14.2f}{r['z_score']:>10.3f}"
              f"{r['pct_leq_zero']:>10.1f}{r['n_periods_improved']:>14}/5")

    z_values = [r["z_score"] for r in results]
    z_range = max(z_values) - min(z_values)
    z_monotone_increasing = all(z_values[i] <= z_values[i + 1] + 0.05 for i in range(len(z_values) - 1))

    print(f"\nRange z tra moltiplicatori: {min(z_values):.3f} - {max(z_values):.3f} (ampiezza {z_range:.3f})")
    print(f"Progressione approssimativamente monotona crescente: {z_monotone_increasing}")
    print("\nINTERPRETAZIONE (criterio fissato prima del test):")
    if z_monotone_increasing and z_range < 1.0:
        print("  z cresce in modo graduale e coerente col moltiplicatore, nessun salto isolato —")
        print("  pattern coerente con un edge sottostante reale (anche se nessun valore supera 2.0,")
        print("  o solo i valori più alti la superano). NON rumore centrato per caso.")
    elif z_range >= 1.5 and not z_monotone_increasing:
        print("  z non progredisce in modo ordinato, variazioni ampie e non lineari tra moltiplicatori —")
        print("  indizio di rumore centrato su pochi trade/giorni anomali, non un edge diffuso.")
        print("  Il filone andrebbe considerato chiuso indipendentemente dal valore massimo di z.")
    else:
        print("  Quadro intermedio, non ricade chiaramente in nessuno dei due casi dichiarati sopra —")
        print("  richiede lettura manuale dei dettagli per periodo prima di una decisione.")


if __name__ == "__main__":
    main()
