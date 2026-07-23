"""
test_ftse_dynamic_exit_combined.py — v3: sostituisce avg_slope (media
cumulata da inizio trade, dimostrata troppo lenta a reagire) con
slope_locale (pendenza sulle ultime 2 barre), mantenendo confronto
RELATIVO contro early_slope e conferma a CONFIRM_BARS=2 barre
consecutive introdotta in v2.

MOTIVAZIONE DEL CAMBIO v2->v3 (analisi su research_v6_trade_path_continuous,
23/07/2026, dopo il risultato ancora negativo di v2, z=-3.15 aggregato):
  - avg_slope = (adx_ora - adx_entrata)/barre_trascorse dilata sempre di
    più la finestra di calcolo man mano che il trade invecchia — ogni
    nuova barra pesa meno, quindi la metrica reagisce lentamente a un
    vero cambio di passo recente
  - slope_locale = (adx_ora - adx_2_barre_fa)/2, confrontata comunque in
    modo RELATIVO contro early_slope (non contro zero in assoluto, che
    si è verificato peggiore), cattura il 45% in più di perdenti veri a
    parità di falsi allarmi sui vincenti (recall 18,8% vs 13,0%,
    selettività quasi identica 2,54:1 vs 2,6:1) — verificato con query
    dedicata prima di ricostruire il test causale.

Storia dei tentativi precedenti su questa famiglia (tutti con motore
vero + bootstrap):
  v1 (trigger singolo, avg_slope): z=-3.28 aggregato, z=-1.90 holdout
  v2 (conferma 2 barre, avg_slope): z=-3.15 aggregato, z=-1.35 holdout
  v3 (conferma 2 barre, slope_locale): QUESTO TEST

Regola unificata per DAX+FTSE100 NELLO STESSO RUN (non due run
separati) — condividono tetto posizioni e kill switch, un run a
singolo strumento perderebbe l'effetto a cascata sugli slot condivisi.

ATTENZIONE — COMPROMESSO DICHIARATO: soglie (NEG_THRESHOLD,
POS_THRESHOLD, DECEL_RATIO) derivate dall'analisi su FTSE100, applicate
identiche a DAX (pattern descrittivo verificato simmetrico tra i due,
vedi sessione). Conteggio uscite riportato per strumento separatamente.

REGOLA (controllata OGNI barra da bar_offset>=3, mentre la posizione è
aperta):
  early_slope   = pendenza ADX barre 0-2 (calcolata una volta, congelata)
  slope_locale  = (adx_ora - adx_2_barre_fa) / 2
  decelerazione = slope_locale < early_slope * DECEL_RATIO

  Se decelerazione E R_corrente <= NEG_THRESHOLD per CONFIRM_BARS barre
  consecutive:
      RAMO A -> chiusura immediata a mercato
  Se decelerazione E R_corrente >= POS_THRESHOLD per CONFIRM_BARS barre
  consecutive:
      RAMO B -> stop dinamico bloccato a LOCK_FRACTION * R_corrente
                (si muove SOLO a favore, mai indietro)
  Il contatore consecutivo si azzera se la condizione smette di essere
  vera anche per una sola barra (nessun reset "a rimbalzo" — verificato
  non discriminante).
  Altrimenti: nessuna modifica, stop/target originali intatti.

PARAMETRI FISSATI ORA, PRIMA DI VEDERE RISULTATI:
  NEG_THRESHOLD = -0.2   (coerente con l'analisi a barra 4)
  POS_THRESHOLD = +0.3   (coerente con l'analisi giveback originale)
  LOCK_FRACTION = 0.5    (blocca il 50% del guadagno corrente)
  DECEL_RATIO   = 0.3    (stessa soglia usata nel controllo di
                          selettività: avg_slope sotto il 30%
                          dell'early_slope = decelerazione)
  MIN_BARS_EARLY = 3     (servono barre 0,1,2 per calcolare early_slope)

La regola oggi NON è condizionata al regime composito, agisce sempre
quando le condizioni scattano, indipendentemente da stress/normale.

SANITY CHECK OBBLIGATORIO: con force_neutral=True, nessuna logica
dinamica applicata, deve riprodurre ESATTAMENTE
BacktestEngineFloatingKillSwitch.

Holdout dichiarato da riga di comando FIN DALL'INIZIO. Successo:
z>=2.0 sull'holdout isolato.
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

NEG_THRESHOLD = -0.2
POS_THRESHOLD = 0.3
LOCK_FRACTION = 0.5
DECEL_RATIO = 0.3
MIN_BARS_EARLY = 3
CONFIRM_BARS = 2  # barre consecutive richieste prima di agire (Ramo A e Ramo B)

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}


def slope_ols(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


class BacktestEngineDynamicExitCombined(BacktestEngineFloatingKillSwitch):

    def __init__(self, capital0, force_neutral: bool = False, **kwargs):
        super().__init__(capital0, **kwargs)
        self.force_neutral = force_neutral

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        if self.open_positions and self.open_positions[-1].instrument == instrument:
            pos = self.open_positions[-1]
            pos.adx_history = [adx_at_entry]
            pos.early_slope = None
            pos.locked_r = None
            pos.neg_streak = 0  # barre consecutive con condizione Ramo A vera
            pos.pos_streak = 0  # barre consecutive con condizione Ramo B vera

    def _apply_dynamic_rule(self, pos, bar, bar_offset, inst) -> bool:
        adx_now = bar["adx"]
        if pd.isna(adx_now):
            return False
        pos.adx_history.append(adx_now)

        if bar_offset == MIN_BARS_EARLY - 1:
            pos.early_slope = slope_ols(list(range(len(pos.adx_history))), pos.adx_history)

        if bar_offset < MIN_BARS_EARLY or pos.early_slope is None or pos.early_slope == 0:
            return False

        # slope_locale invece di avg_slope: pendenza sulle ultime 2 barre
        # (non media cumulata da inizio trade) — verificato con analisi
        # su research_v6_trade_path_continuous che cattura il 45% in più
        # di perdenti veri a parità di falsi allarmi sui vincenti (18,8%
        # vs 13,0% recall, rapporto selettività quasi identico 2,54:1 vs
        # 2,6:1). avg_slope cumulato diluisce troppo lentamente lo scatto
        # iniziale, slope_locale reagisce al cambio di passo recente.
        if len(pos.adx_history) < 3:
            return False
        slope_locale = (pos.adx_history[-1] - pos.adx_history[-3]) / 2.0
        is_decel = slope_locale < pos.early_slope * DECEL_RATIO
        if not is_decel:
            pos.neg_streak = 0
            pos.pos_streak = 0
            return False

        stop_distance = pos.atr_at_entry * inst.atr_multiplier
        close_price = bar["close"]
        if pos.direction == "long":
            r_now = (close_price - pos.entry_price) / stop_distance
        else:
            r_now = (pos.entry_price - close_price) / stop_distance

        # Condizioni valutate ogni barra, ma l'azione scatta solo dopo
        # CONFIRM_BARS barre CONSECUTIVE con la stessa condizione vera —
        # verificato con analisi su research_v6_trade_path_continuous: il
        # trigger al primo tocco causava 65% di falsi positivi sul Ramo B
        # (il prezzo non scendeva mai al livello bloccato) e colpiva il
        # 13,8% dei vincenti anche sul Ramo A (tuffo iniziale normale).
        if r_now <= NEG_THRESHOLD:
            pos.neg_streak += 1
            pos.pos_streak = 0
        elif r_now >= POS_THRESHOLD:
            pos.pos_streak += 1
            pos.neg_streak = 0
        else:
            pos.neg_streak = 0
            pos.pos_streak = 0

        if pos.neg_streak >= CONFIRM_BARS:
            spread = inst.spread_fixed
            exit_price = close_price - spread / 2 if pos.direction == "long" else close_price + spread / 2
            self._close_position(pos, bar["timestamp"], exit_price, "dynamic_exit_negative")
            return True

        if pos.pos_streak >= CONFIRM_BARS:
            target_locked_r = r_now * LOCK_FRACTION
            if pos.locked_r is None or target_locked_r > pos.locked_r:
                pos.locked_r = target_locked_r
                if pos.direction == "long":
                    new_stop = pos.entry_price + target_locked_r * stop_distance
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop
                else:
                    new_stop = pos.entry_price - target_locked_r * stop_distance
                    if new_stop < pos.stop_loss:
                        pos.stop_loss = new_stop
        return False

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
                inst = self.instruments[pos.instrument]

                if not self.force_neutral:
                    bar_offset = bar_index - pos.entry_bar_index
                    closed_here = self._apply_dynamic_rule(pos, bar, bar_offset, inst)
                    if closed_here:
                        continue

                self._try_close_position(pos, bar, bar_index, inst)

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


def run_period_dynamic(signals_by_instrument, start, end, force_neutral=False):
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineDynamicExitCombined(capital0=CAPITAL_V6, force_neutral=force_neutral)
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


def bootstrap_periods(signals, period_labels):
    all_delta_days = []
    period_summary = []
    for period_name in period_labels:
        start, end = PERIODS[period_name]
        baseline_trades = run_period_baseline(signals, start, end)
        dyn_trades = run_period_dynamic(signals, start, end)

        baseline_pnl = float(baseline_trades["pnl"].sum()) if len(baseline_trades) else 0.0
        dyn_pnl = float(dyn_trades["pnl"].sum()) if len(dyn_trades) else 0.0

        d_baseline = daily_pnl(baseline_trades, start, end)
        d_dyn = daily_pnl(dyn_trades, start, end)
        all_delta_days.append(d_dyn - d_baseline)

        n_dynamic_exit = int((dyn_trades["exit_reason"] == "dynamic_exit_negative").sum()) if len(dyn_trades) else 0

        baseline_exit_counts = baseline_trades["exit_reason"].value_counts().to_dict() if len(baseline_trades) else {}
        dyn_exit_counts = dyn_trades["exit_reason"].value_counts().to_dict() if len(dyn_trades) else {}

        per_instrument = {}
        for inst_name in ("DAX", "FTSE100"):
            b_inst = baseline_trades[baseline_trades["instrument"] == inst_name] if len(baseline_trades) else baseline_trades
            d_inst = dyn_trades[dyn_trades["instrument"] == inst_name] if len(dyn_trades) else dyn_trades
            b_pnl_inst = float(b_inst["pnl"].sum()) if len(b_inst) else 0.0
            d_pnl_inst = float(d_inst["pnl"].sum()) if len(d_inst) else 0.0
            n_exit_inst = int((d_inst["exit_reason"] == "dynamic_exit_negative").sum()) if len(d_inst) else 0
            b_exit_inst = b_inst["exit_reason"].value_counts().to_dict() if len(b_inst) else {}
            d_exit_inst = d_inst["exit_reason"].value_counts().to_dict() if len(d_inst) else {}
            per_instrument[inst_name] = {
                "delta": d_pnl_inst - b_pnl_inst, "n_dynamic_exit": n_exit_inst,
                "baseline_trades": len(b_inst), "dyn_trades": len(d_inst),
                "baseline_exit_counts": b_exit_inst, "dyn_exit_counts": d_exit_inst,
            }

        period_summary.append({
            "period": period_name, "baseline_pnl": baseline_pnl, "dyn_pnl": dyn_pnl,
            "delta": dyn_pnl - baseline_pnl,
            "baseline_trades": len(baseline_trades), "dyn_trades": len(dyn_trades),
            "n_dynamic_exit_negativo": n_dynamic_exit,
            "baseline_exit_counts": baseline_exit_counts, "dyn_exit_counts": dyn_exit_counts,
            "per_instrument": per_instrument,
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
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    for s in res["period_summary"]:
        print(f"  {s['period']:<12} trade base={s['baseline_trades']:>4} dyn={s['dyn_trades']:>4}  "
              f"(uscite negative anticipate={s['n_dynamic_exit_negativo']:>3})  "
              f"PnL base={s['baseline_pnl']:>10.2f}  PnL dyn={s['dyn_pnl']:>10.2f}  "
              f"delta={s['delta']:>+9.2f}")
        print(f"      exit_reason baseline: {s['baseline_exit_counts']}")
        print(f"      exit_reason dinamico: {s['dyn_exit_counts']}")
        for inst_name, pi in s["per_instrument"].items():
            print(f"      {inst_name:<8} delta={pi['delta']:>+9.2f}  "
                  f"uscite anticipate={pi['n_dynamic_exit']:>3}  "
                  f"trade base={pi['baseline_trades']:>4} dyn={pi['dyn_trades']:>4}")
            print(f"          exit_reason base: {pi['baseline_exit_counts']}")
            print(f"          exit_reason dyn:  {pi['dyn_exit_counts']}")
    print(f"\n  Delta osservato: {res['observed_delta']:+.2f} EUR")
    print(f"  Z-score: {res['z_score']:.3f}")
    print(f"  %% iterazioni con delta<=0: {res['pct_leq_zero']:.1f}%%")
    print(f"  95%% CI bootstrap: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")


def sanity_check(signals):
    print("=== SANITY CHECK (obbligatorio) ===")
    start, end = PERIODS["2015-2016"]
    baseline = run_period_baseline(signals, start, end)
    neutral = run_period_dynamic(signals, start, end, force_neutral=True)
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

    print(f"\nv3 — slope_locale (2 barre) invece di avg_slope cumulato. Parametri: "
          f"NEG_THRESHOLD={NEG_THRESHOLD}, POS_THRESHOLD={POS_THRESHOLD}, "
          f"LOCK_FRACTION={LOCK_FRACTION}, DECEL_RATIO={DECEL_RATIO}, CONFIRM_BARS={CONFIRM_BARS}\n")

    sanity_check(signals)

    res_holdout_only = bootstrap_periods(signals, [holdout_label])
    print_result(f"TEST A — SOLO HOLDOUT ({holdout_label})", res_holdout_only)

    res_all = bootstrap_periods(signals, list(PERIODS.keys()))
    print_result("TEST B — TUTTI I 5 PERIODI (di contesto)", res_all)

    print("\n" + "=" * 90)
    print(f"VERDETTO — holdout {holdout_label} (criterio: z>=2.0 su Test A)")
    print("=" * 90)
    print(f"Test A (solo holdout, IL TEST CHE CONTA): z={res_holdout_only['z_score']:.3f}")
    print(f"Test B (5 periodi, di contesto): z={res_all['z_score']:.3f}")
    if res_holdout_only["z_score"] >= 2.0:
        print("Supera la soglia sull'holdout isolato.")
    else:
        print("Non supera la soglia sull'holdout isolato.")


if __name__ == "__main__":
    main()
