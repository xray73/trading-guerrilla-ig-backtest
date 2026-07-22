"""
test_ftse_composite_size_boost.py — Primo test causale sul regime
composito (corr_dax_ftse_7d alta + atr_pct alta = "stress") applicato
SOLO a FTSE100, con SIZE BOOST invece di riduzione — direzione opposta
a tutti i tentativi precedenti sulla correlazione isolata (5 varianti
bocciate ieri, tutte con riduzione/skip).

MOTIVAZIONE (analisi descrittiva 22/07/2026, via query su
research_v6_trade_features + market_regime_indicators): win rate
FTSE100 in stato "stress" (entrambe le condizioni) = 39,4% (n=254),
MIGLIORE del baseline 36,3% (n=509) — pattern consistente su 4/5
periodi ufficiali (eccezione: 2020-covid, dove va nella direzione
opposta). Analisi a quadranti mostra un'interazione non additiva per
FTSE100 (ATR% alto da solo peggiora, corr alta da sola è neutra, la
COMBINAZIONE migliora) — diverso da DAX, dove la correlazione da sola
guida il pattern (motivo per cui qui NON tocchiamo DAX, sarebbe una
variante del filone già bocciato).

CRITERI FISSATI PRIMA DI VEDERE I RISULTATI DI QUESTO SCRIPT:
  - Moltiplicatore size: 1.3x quando stato=stress su FTSE100, 1.0x
    altrimenti. Valore dichiarato ora (aumento moderato, non ottimizzato
    sui dati), non ricalibrato dopo aver visto l'output.
  - Soglie stato: atr_pct FTSE100 > 0.2031323100223204 (terzile alto,
    da NTILE(3) su market_regime_indicators) E corr_dax_ftse_7d >
    0.7853464827260775 (idem, terzile alto) — stesse soglie usate
    nell'analisi descrittiva che ha generato l'ipotesi.
  - Successo: delta positivo con bootstrap z>=2 (soglia standard del
    progetto) sui 5 periodi ufficiali combinati. Se non regge, si
    chiude come le altre 5 varianti sulla correlazione.

SANITY CHECK OBBLIGATORIO (principio non negoziabile del progetto):
con moltiplicatore forzato a 1.0 su TUTTI i trade (FORCE_NEUTRAL=True),
questo motore deve riprodurre ESATTAMENTE gli stessi trade/PnL di
BacktestEngineFloatingKillSwitch — verificato PRIMA del test vero,
stampato a schermo, non assunto.
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
BOOST_MULTIPLIER = 1.3
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
    """Size boost SOLO per FTSE100 in stato composito 'stress'. DAX
    invariato in ogni condizione (moltiplicatore sempre 1.0). Con
    force_neutral=True, moltiplicatore sempre 1.0 anche per FTSE100 —
    usato SOLO per il sanity check, mai per il test vero."""

    def __init__(self, capital0, stress_lookup: pd.Series, force_neutral: bool = False, **kwargs):
        super().__init__(capital0, **kwargs)
        self.stress_lookup = stress_lookup
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
                multiplier = BOOST_MULTIPLIER
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


def run_period_boost(signals_by_instrument, start, end, stress_lookup, force_neutral=False):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFtseCompositeBoost(
        capital0=CAPITAL_V6, stress_lookup=stress_lookup, force_neutral=force_neutral)
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
    print("=== SANITY CHECK (obbligatorio, PRIMA del test vero) ===")
    print("Confronto BacktestEngineFloatingKillSwitch vs BacktestEngineFtseCompositeBoost "
          "(force_neutral=True) sul primo periodo (2015-2016) — devono essere IDENTICI.\n")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_boost(signals, start, end, stress_lookup, force_neutral=True)

    n_base, n_neutral = len(baseline), len(neutral)
    pnl_base = float(baseline["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral["pnl"].sum()) if n_neutral else 0.0

    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral (force_neutral=True): {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")

    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("\n  *** SANITY CHECK FALLITO *** — le due versioni non sono identiche a parametri "
              "neutri. INTERROMPO, non procedo col test causale finché non è corretto.")
        sys.exit(1)
    print("\n  Sanity check OK — procedo col test causale.\n")


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
          f"({100*stress_lookup.mean():.1f}%)")

    sanity_check(signals, stress_lookup)

    all_delta_days = []
    period_summary = []

    for period_name, (start, end) in PERIODS.items():
        print(f"\n=== Periodo {period_name} ===")
        baseline_trades = run_period_baseline(signals, start, end)
        boost_trades = run_period_boost(signals, start, end, stress_lookup)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        boost_pnl = float(boost_trades["pnl"].sum()) if len(boost_trades) else 0.0
        delta_pnl = boost_pnl - baseline_pnl

        print(f"  Baseline: {len(baseline_trades)} trade, PnL {baseline_pnl:+.2f} EUR")
        print(f"  Boost FTSE100/stress: {len(boost_trades)} trade, PnL {boost_pnl:+.2f} EUR")
        print(f"  Delta: {delta_pnl:+.2f} EUR")

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_boost = daily_pnl(boost_trades, start, end)
        delta_series = (d_boost - d_baseline)
        all_delta_days.append(delta_series)

        period_summary.append({"period": period_name, "baseline_pnl": baseline_pnl,
                                "boost_pnl": boost_pnl, "delta": delta_pnl,
                                "baseline_trades": len(baseline_trades), "boost_trades": len(boost_trades)})

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
    print(f"{'Periodo':<12}{'Trade base':>12}{'Trade boost':>12}{'PnL base':>14}{'PnL boost':>14}{'Delta':>12}")
    for s in period_summary:
        print(f"{s['period']:<12}{s['baseline_trades']:>12}{s['boost_trades']:>12}"
              f"{s['baseline_pnl']:>14.2f}{s['boost_pnl']:>14.2f}{s['delta']:>12.2f}")

    total_baseline = sum(s["baseline_pnl"] for s in period_summary)
    total_boost = sum(s["boost_pnl"] for s in period_summary)
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)
    print(f"\nPnL totale baseline (5 periodi): {total_baseline:+.2f} EUR")
    print(f"PnL totale boost (5 periodi): {total_boost:+.2f} EUR")
    print(f"Periodi migliorati: {n_periods_improved}/5")

    print("\n=== VERDETTO (criterio fissato PRIMA del test: z>=2 per adozione) ===")
    if z_score >= 2.0:
        print(f"z={z_score:.2f} >= 2.0 — segnale supera la soglia standard del progetto.")
    else:
        print(f"z={z_score:.2f} < 2.0 — non supera la soglia, coerente con lo standard "
              f"applicato a tutte le altre 5 varianti sulla correlazione (tutte bocciate).")


if __name__ == "__main__":
    main()
