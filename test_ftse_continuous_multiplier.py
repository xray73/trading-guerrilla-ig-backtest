"""
test_ftse_continuous_multiplier.py — Idea 2 (segnata in memoria
22/07/2026): invece di un moltiplicatore fisso attivato da soglie
binarie a terzili (test_ftse_composite_size_boost.py /
_sweep.py, z massimo 2,001 a 1.7x, effetto debole ma non spiegabile da
artefatti di composizione — vedi diagnose_boost_composition.py),
prova una funzione CONTINUA che cattura la forma reale dell'interazione
non additiva trovata nell'analisi a quadranti (ATR% alto da solo
peggiora, corr alta da sola e' neutra, ENTRAMBI insieme migliorano).

FUNZIONE DICHIARATA ORA, PRIMA DI VEDERE RISULTATI:
  a(t) = rango percentile di atr_pct (FTSE100) nella distribuzione
         storica completa (0=minimo mai visto, 1=massimo mai visto)
  c(t) = rango percentile di corr_dax_ftse_7d nella distribuzione
         storica completa
  moltiplicatore(t) = 1.0 + a(t) * c(t)        [range: 1.0 -> 2.0]

Interazione MOLTIPLICATIVA (non additiva): se uno dei due fattori e'
basso, il prodotto resta vicino a zero anche se l'altro e' alto —
coerente con "serve la combinazione", non un singolo ingrediente.
Applicato SOLO a FTSE100 (DAX invariato, stessa scelta di design delle
versioni precedenti — l'asimmetria DAX/FTSE100 e' specifica per
strumento).

ONESTA' METODOLOGICA: la forma della funzione (moltiplicativa, non
additiva) e' ispirata dal pattern gia' visto nel quadrante — non e'
un'ipotesi generata alla cieca. I percentili sono calcolati su TUTTA
la storia disponibile (stesso approccio "fit su train, nessun holdout
separato" gia' usato per le soglie terzili in questo filone — non e'
un nuovo standard piu' permissivo, e' coerenza col resto del filone).

SANITY CHECK OBBLIGATORIO: con force_neutral=True (moltiplicatore
sempre 1.0), deve riprodurre ESATTAMENTE BacktestEngineFloatingKillSwitch.

Successo dichiarato: z>=2.0 sul bootstrap a blocchi di giornata, 5
periodi ufficiali combinati — stessa soglia di ogni altro test in
questo progetto.
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


class BacktestEngineFtseContinuousMultiplier(BacktestEngineFloatingKillSwitch):
    """Moltiplicatore continuo 1.0 + a(t)*c(t) per FTSE100. Con
    force_neutral=True, moltiplicatore sempre 1.0 — solo per sanity check."""

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


def sanity_check(signals, multiplier_lookup):
    print("=== SANITY CHECK (obbligatorio, PRIMA del test vero) ===")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_continuous(signals, start, end, multiplier_lookup, force_neutral=True)

    n_base, n_neutral = len(baseline), len(neutral)
    pnl_base = float(baseline["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral["pnl"].sum()) if n_neutral else 0.0

    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral (force_neutral=True): {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")

    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("\n  *** SANITY CHECK FALLITO *** — INTERROMPO.")
        sys.exit(1)
    print("  Sanity check OK — procedo col test causale.\n")


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico atr_pct (FTSE100) e corr_dax_ftse_7d, costruisco moltiplicatore continuo...")
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

    # a(t), c(t): rango percentile su TUTTA la distribuzione storica disponibile
    a = combined["atr"].rank(pct=True)
    c = combined["corr"].rank(pct=True)
    multiplier_lookup = 1.0 + (a * c)

    print(f"  {len(multiplier_lookup)} punti. Moltiplicatore: "
          f"min={multiplier_lookup.min():.3f} max={multiplier_lookup.max():.3f} "
          f"media={multiplier_lookup.mean():.3f} mediana={multiplier_lookup.median():.3f}")

    sanity_check(signals, multiplier_lookup)

    all_delta_days = []
    period_summary = []

    for period_name, (start, end) in PERIODS.items():
        print(f"\n=== Periodo {period_name} ===")
        baseline_trades = run_period_baseline(signals, start, end)
        cont_trades = run_period_continuous(signals, start, end, multiplier_lookup)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        cont_pnl = float(cont_trades["pnl"].sum()) if len(cont_trades) else 0.0
        delta_pnl = cont_pnl - baseline_pnl

        print(f"  Baseline: {len(baseline_trades)} trade, PnL {baseline_pnl:+.2f} EUR")
        print(f"  Continuo: {len(cont_trades)} trade, PnL {cont_pnl:+.2f} EUR")
        print(f"  Delta: {delta_pnl:+.2f} EUR")

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_cont = daily_pnl(cont_trades, start, end)
        delta_series = (d_cont - d_baseline)
        all_delta_days.append(delta_series)

        period_summary.append({"period": period_name, "baseline_pnl": baseline_pnl,
                                "cont_pnl": cont_pnl, "delta": delta_pnl,
                                "baseline_trades": len(baseline_trades), "cont_trades": len(cont_trades)})

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
    print(f"{'Periodo':<12}{'Trade base':>12}{'Trade cont':>12}{'PnL base':>14}{'PnL cont':>14}{'Delta':>12}")
    for s in period_summary:
        print(f"{s['period']:<12}{s['baseline_trades']:>12}{s['cont_trades']:>12}"
              f"{s['baseline_pnl']:>14.2f}{s['cont_pnl']:>14.2f}{s['delta']:>12.2f}")

    total_baseline = sum(s["baseline_pnl"] for s in period_summary)
    total_cont = sum(s["cont_pnl"] for s in period_summary)
    n_periods_improved = sum(1 for s in period_summary if s["delta"] > 0)
    print(f"\nPnL totale baseline (5 periodi): {total_baseline:+.2f} EUR")
    print(f"PnL totale continuo (5 periodi): {total_cont:+.2f} EUR")
    print(f"Periodi migliorati: {n_periods_improved}/5")

    print("\n=== VERDETTO (criterio fissato PRIMA del test: z>=2 per adozione) ===")
    if z_score >= 2.0:
        print(f"z={z_score:.2f} >= 2.0 — supera la soglia standard del progetto.")
    else:
        print(f"z={z_score:.2f} < 2.0 — non supera la soglia.")


if __name__ == "__main__":
    main()
