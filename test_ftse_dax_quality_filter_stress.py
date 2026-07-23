"""
test_ftse_dax_quality_filter_stress.py — Idea 2: invece di modulare la
size (filone chiuso oggi), in stato stress alza la soglia di qualità
richiesta per APRIRE il trade — scarta i candidati nel terzile peggiore
di breakout_distance_pts (distanza di chiusura oltre il livello di
breakout: close-rolling_high per long, rolling_low-close per short).

MOTIVAZIONE (analisi descrittiva 22/07/2026): in condizioni normali il
terzile di breakout_distance discrimina poco (spread R quasi nullo su
DAX, debole su FTSE100). In stress diventa molto più discriminante su
ENTRAMBI gli strumenti (DAX: terzile peggiore passa da +0,273R a
-0,116R; FTSE100: da -0,024R a +0,067R ma con spread interno molto più
ampio) — a differenza del size-boost, qui il pattern è SIMMETRICO tra i
due strumenti, quindi il filtro si applica a DAX E FTSE100.

LEZIONE APPLICATA FIN DALL'INIZIO (non aggiunta dopo, come nel test
precedente): sia le soglie di stato (ATR%/corr terzile alto) SIA la
soglia di qualità (breakout_distance terzile peggiore) sono fit SOLO
sul train (esclude il periodo holdout), applicate COSÌ COME SONO
all'holdout, senza ricalibrare. Holdout parametrizzabile da riga di
comando (stesso principio di doppia validazione incrociata già usato
per corr_ftse_gold e per il moltiplicatore continuo, entrambi chiusi
oggi per non aver retto a QUESTO stesso controllo).

MECCANISMO: il candidato viene scartato interamente (nessun trade
aperto, non è size=0 né size ridotta) se, al momento del segnale:
  - stato composito = stress (ATR% terzile alto E corr terzile alto,
    soglie fit su train, PER STRUMENTO per ATR%, condivisa per corr)
  - breakout_distance_pts >= soglia terzile peggiore fit su train
    (PER STRUMENTO — DAX e FTSE100 hanno scale di prezzo diverse)

Uso: python test_ftse_dax_quality_filter_stress.py 2020-covid
     python test_ftse_dax_quality_filter_stress.py 2015-2016

Sanity check obbligatorio: con filtro disattivato (force_neutral=True),
riproduce ESATTAMENTE BacktestEngineFloatingKillSwitch.
Successo dichiarato: z>=2.0 sull'holdout isolato (Test A) — non solo
sull'aggregato con soglie fit-su-train (Test B), lezione del test
precedente.
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


class BacktestEngineQualityFilterStress(BacktestEngineFloatingKillSwitch):
    """Scarta interamente i candidati nel terzile peggiore di
    breakout_distance QUANDO lo stato composito e' stress. Tutte le
    soglie sono passate dall'esterno, gia' fit-su-train."""

    def __init__(self, capital0, atr_lookup: dict[str, pd.Series], corr_lookup: pd.Series,
                 atr_thresh: dict[str, float], corr_thresh: float,
                 quality_thresh: dict[str, float], force_neutral: bool = False, **kwargs):
        super().__init__(capital0, **kwargs)
        self.atr_lookup = atr_lookup
        self.corr_lookup = corr_lookup
        self.atr_thresh = atr_thresh
        self.corr_thresh = corr_thresh
        self.quality_thresh = quality_thresh
        self.force_neutral = force_neutral

    def _lookup_value(self, series: pd.Series, ts) -> float:
        if series is None or series.empty:
            return float("nan")
        idx = series.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return float("nan")
        return float(series.iloc[idx])

    def _is_stress(self, instrument: str, ts) -> bool:
        atr_val = self._lookup_value(self.atr_lookup.get(instrument), ts)
        corr_val = self._lookup_value(self.corr_lookup, ts)
        if pd.isna(atr_val) or pd.isna(corr_val):
            return False
        return atr_val > self.atr_thresh[instrument] and corr_val > self.corr_thresh

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        tradable_instruments = [
            name for name in data
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")

        all_timestamps = sorted(set().union(
            *[set(data[i]["timestamp"]) for i in tradable_instruments]))

        for ts in all_timestamps:
            self._reset_day_if_needed(ts)

            for pos in list(self.open_positions):
                inst_df = data[pos.instrument]
                row = inst_df.loc[inst_df["timestamp"] == ts]
                if row.empty:
                    continue
                bar = row.iloc[0]
                bar_index = row.index[0]
                self._try_close_position(pos, bar, bar_index, self.instruments[pos.instrument])

            self.equity_curve.append((ts, self.capital))

            if not self._kill_switch_active and self.open_positions:
                current_bars = {}
                for pos in self.open_positions:
                    inst_df = data[pos.instrument]
                    row = inst_df.loc[inst_df["timestamp"] == ts]
                    if not row.empty:
                        current_bars[pos.instrument] = row.iloc[0]
                perdita_pct = self._floating_loss_pct(current_bars)
                if perdita_pct >= self.p.kill_switch_pct:
                    self._kill_switch_active = True

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            candidates = []
            for name in tradable_instruments:
                inst_df = data[name]
                idx = inst_df.index[inst_df["timestamp"] == ts]
                if len(idx) == 0:
                    continue
                i = idx[0]
                if i == 0:
                    continue
                prev_bar = inst_df.iloc[i - 1]
                cur_bar = inst_df.iloc[i]
                if prev_bar["signal"] not in ("long", "short"):
                    continue
                already_open = any(p.instrument == name for p in self.open_positions)
                if already_open:
                    continue

                if not self.force_neutral:
                    if prev_bar["signal"] == "long":
                        breakout_distance = prev_bar["close"] - prev_bar["rolling_high"]
                    else:
                        breakout_distance = prev_bar["rolling_low"] - prev_bar["close"]

                    if self._is_stress(name, prev_bar["timestamp"]):
                        thresh = self.quality_thresh.get(name)
                        if thresh is not None and breakout_distance >= thresh:
                            continue

                candidates.append({
                    "instrument": name, "direction": prev_bar["signal"],
                    "bar": cur_bar, "atr": prev_bar["atr"], "adx": prev_bar["adx"],
                    "rr": self.p.rr_target,
                })

            if not candidates:
                continue

            candidates.sort(key=lambda c: (-c["rr"], self._correlation_penalty(c["instrument"])))

            slots_free = self.p.max_concurrent_positions - len(self.open_positions)
            for c in candidates:
                if slots_free <= 0:
                    break
                if self._orders_today >= self.p.max_new_orders_per_day:
                    break
                if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                    continue
                self._open_position(c["instrument"], c["direction"], c["bar"],
                                     c["atr"], c["adx"])
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


def slice_period(signals, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def run_period_baseline(signals_by_instrument, start, end):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    trades_df, _ = engine_.run(sliced)
    return trades_df


def run_period_filtered(signals_by_instrument, start, end, atr_lookup, corr_lookup,
                         atr_thresh, corr_thresh, quality_thresh, force_neutral=False):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineQualityFilterStress(
        capital0=CAPITAL_V6, atr_lookup=atr_lookup, corr_lookup=corr_lookup,
        atr_thresh=atr_thresh, corr_thresh=corr_thresh, quality_thresh=quality_thresh,
        force_neutral=force_neutral)
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


def bootstrap_periods(signals, atr_lookup, corr_lookup, atr_thresh, corr_thresh,
                       quality_thresh, period_labels):
    all_delta_days = []
    period_summary = []
    for period_name in period_labels:
        start, end = PERIODS[period_name]
        baseline_trades = run_period_baseline(signals, start, end)
        filt_trades = run_period_filtered(signals, start, end, atr_lookup, corr_lookup,
                                           atr_thresh, corr_thresh, quality_thresh)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        filt_pnl = float(filt_trades["pnl"].sum()) if len(filt_trades) else 0.0

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_filt = daily_pnl(filt_trades, start, end)
        all_delta_days.append(d_filt - d_baseline)

        period_summary.append({
            "period": period_name, "baseline_pnl": baseline_pnl, "filt_pnl": filt_pnl,
            "delta": filt_pnl - baseline_pnl,
            "baseline_trades": len(baseline_trades), "filt_trades": len(filt_trades),
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
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for s in res["period_summary"]:
        print(f"  {s['period']:<12} trade base={s['baseline_trades']:>4} filt={s['filt_trades']:>4}  "
              f"PnL base={s['baseline_pnl']:>10.2f}  PnL filt={s['filt_pnl']:>10.2f}  "
              f"delta={s['delta']:>+9.2f}")
    print(f"\n  Delta osservato: {res['observed_delta']:+.2f} EUR")
    print(f"  Z-score: {res['z_score']:.3f}")
    print(f"  %% iterazioni con delta<=0: {res['pct_leq_zero']:.1f}%%")
    print(f"  95%% CI bootstrap: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")


def sanity_check(signals, atr_lookup, corr_lookup, atr_thresh, corr_thresh, quality_thresh):
    print("=== SANITY CHECK (obbligatorio) ===")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_filtered(signals, start, end, atr_lookup, corr_lookup,
                                   atr_thresh, corr_thresh, quality_thresh, force_neutral=True)
    n_base, n_neutral = len(baseline), len(neutral)
    pnl_base = float(baseline["pnl"].sum()) if n_base else 0.0
    pnl_neutral = float(neutral["pnl"].sum()) if n_neutral else 0.0
    print(f"  Baseline: {n_base} trade, PnL {pnl_base:+.2f} EUR")
    print(f"  Neutral (force_neutral=True): {n_neutral} trade, PnL {pnl_neutral:+.2f} EUR")
    if n_base != n_neutral or abs(pnl_base - pnl_neutral) > 0.01:
        print("\n  *** SANITY CHECK FALLITO *** — INTERROMPO.")
        sys.exit(1)
    print("  OK\n")


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} PERIODO_HOLDOUT")
        print(f"Periodi disponibili: {', '.join(PERIODS)}")
        sys.exit(1)

    holdout_label = sys.argv[1].strip()
    if holdout_label not in PERIODS:
        print(f"ERRORE: periodo '{holdout_label}' non riconosciuto. Disponibili: {', '.join(PERIODS)}")
        sys.exit(1)

    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico atr_pct (DAX+FTSE100) e corr_dax_ftse_7d...")
    atr_series = {}
    for inst_name in ("DAX", "FTSE100"):
        rows = d1(f"SELECT timestamp, atr_pct FROM market_regime_indicators "
                  f"WHERE instrument='{inst_name}' AND atr_pct IS NOT NULL ORDER BY timestamp ASC")
        atr_series[inst_name] = pd.Series(
            [r["atr_pct"] for r in rows],
            index=pd.to_datetime([r["timestamp"] for r in rows], utc=True))

    rows_corr = d1("SELECT timestamp, corr_dax_ftse_7d FROM market_regime_indicators "
                   "WHERE instrument='DAX' AND corr_dax_ftse_7d IS NOT NULL ORDER BY timestamp ASC")
    corr_series = pd.Series(
        [r["corr_dax_ftse_7d"] for r in rows_corr],
        index=pd.to_datetime([r["timestamp"] for r in rows_corr], utc=True))

    holdout_start, holdout_end = PERIODS[holdout_label]
    holdout_start_ts = pd.Timestamp(holdout_start, tz="UTC")
    holdout_end_ts = pd.Timestamp(holdout_end, tz="UTC") + pd.Timedelta(days=1)

    def train_mask(idx):
        return ~((idx >= holdout_start_ts) & (idx < holdout_end_ts))

    atr_thresh = {}
    for inst_name, series in atr_series.items():
        train_vals = series[train_mask(series.index)]
        atr_thresh[inst_name] = train_vals.quantile(2 / 3)

    corr_train_vals = corr_series[train_mask(corr_series.index)]
    corr_thresh = corr_train_vals.quantile(2 / 3)

    print(f"  Soglie ATR%% terzile alto (fit train): DAX={atr_thresh['DAX']:.4f} "
          f"FTSE100={atr_thresh['FTSE100']:.4f}")
    print(f"  Soglia corr terzile alto (fit train): {corr_thresh:.4f}")

    quality_thresh = {}
    for inst_name in ("DAX", "FTSE100"):
        sig_df = signals[inst_name]
        train_sig = sig_df[train_mask(pd.to_datetime(sig_df["timestamp"], utc=True))]
        long_sig = train_sig[train_sig["signal"] == "long"]
        short_sig = train_sig[train_sig["signal"] == "short"]
        distances = pd.concat([
            long_sig["close"] - long_sig["rolling_high"],
            short_sig["rolling_low"] - short_sig["close"],
        ]).dropna()
        quality_thresh[inst_name] = distances.quantile(2 / 3)
        print(f"  Soglia qualita' (terzile peggiore breakout_distance, fit train) {inst_name}: "
              f"{quality_thresh[inst_name]:.2f} pt (n segnali train={len(distances)})")

    print()
    sanity_check(signals, atr_series, corr_series, atr_thresh, corr_thresh, quality_thresh)

    res_holdout_only = bootstrap_periods(signals, atr_series, corr_series, atr_thresh, corr_thresh,
                                          quality_thresh, [holdout_label])
    print_result(f"TEST A — SOLO HOLDOUT ({holdout_label}), tutte le soglie fit su train", res_holdout_only)

    res_all_trainfit = bootstrap_periods(signals, atr_series, corr_series, atr_thresh, corr_thresh,
                                          quality_thresh, list(PERIODS.keys()))
    print_result(f"TEST B — TUTTI I 5 PERIODI, soglie fit-su-train (holdout={holdout_label})",
                  res_all_trainfit)

    print("\n" + "=" * 78)
    print(f"VERDETTO — holdout {holdout_label} (criterio: z>=2.0 su Test A, non solo Test B)")
    print("=" * 78)
    print(f"Test A (solo holdout, IL TEST CHE CONTA): z={res_holdout_only['z_score']:.3f}")
    print(f"Test B (5 periodi, fit-su-train, di contesto): z={res_all_trainfit['z_score']:.3f}")
    if res_holdout_only["z_score"] >= 2.0:
        print("Supera la soglia sull'holdout isolato — primo caso in questo filone a riuscirci.")
    else:
        print("Non supera la soglia sull'holdout isolato — stesso esito del size-boost, "
              "coerente con lo standard applicato a tutti gli altri tentativi.")


if __name__ == "__main__":
    main()
