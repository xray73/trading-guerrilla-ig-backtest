"""
test_adx_accel_exit.py — TEST 6/6 del programma di validazione (21/07/2026).

Domanda: chiudere anticipatamente alla barra 4 un trade la cui
ACCELERAZIONE dell'ADX (variazione della pendenza, non la pendenza
stessa) scende sotto una soglia negativa migliora il risultato?

Origine: analisi bar-per-bar (velocita'/accelerazione di prezzo e ADX,
richiesta in chat) sul dataset di ricerca gia' esistente — nessun nuovo
backtest per l'esplorazione. L'accelerazione ADX (adx[N] - 2*adx[N-1] +
adx[N-2]) mostra una separazione vincitori/perdenti molto piu' netta e
STABILE della pendenza semplice: i perdenti decelerano 5-6x piu' in
fretta dei vincitori, differenza presente e coerente dalla barra 2 alla
8. Conto grezzo (somma price_r - r_multiple sui trade sotto soglia,
STESSA metodologia gia' validata per il test 4): positivo su TUTTE le
5 soglie testate (-0.1 a -0.8) alla barra 4, unica barra a comportarsi
cosi' su tutta la griglia — le altre barre (2,3,5,6,8) sono quasi
sempre negative, stesso pattern degli altri test falliti.

DUE varianti in parallelo, nessuna scelta dopo aver visto i risultati:
  A) check_bar=4, adx_accel_threshold=-0.2 (n=468, swing grezzo +30.3)
  B) check_bar=4, adx_accel_threshold=-0.5 (n=320, swing grezzo +29.9)

Regola: la storia dei valori ADX di ogni posizione viene tracciata
bar-per-bar (dizionario interno, MAI nella dataclass Position di
engine.py). Alla barra 4 esatta dall'ingresso, se l'accelerazione
(adx[4]-2*adx[3]+adx[2]) e' sotto soglia, chiude subito
(exit_reason='adx_decel_exit').

BOOTSTRAP A RISCHIO FISSO (versione corretta, standard da questa
sessione in poi). Motore: nuova sottoclasse isolata di
BacktestEngineFloatingKillSwitch. SANITY CHECK: adx_accel_threshold=None
deve produrre risultati IDENTICI al motore standard.

Nessuna scrittura su trades/backtest_runs/live_*/research_v6_*. Output
SOLO aggregati.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
VARIANTS = [
    ("A_th-0.2", 4, -0.2),
    ("B_th-0.5", 4, -0.5),
]
N_BOOT = 2000
SEED = 20260721
KILL_SWITCH_PCT = eng.PARAMS.kill_switch_pct

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", None),
]


# =====================================================================
# 1) MOTORE — sottoclasse isolata, uscita per decelerazione ADX
# =====================================================================

class BacktestEngineADXAccelExit(BacktestEngineFloatingKillSwitch):
    def __init__(self, *args, check_bar: int = None, adx_accel_threshold: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_bar = check_bar
        self.adx_accel_threshold = adx_accel_threshold
        self._adx_history: dict[int, list] = {}

    def _update_and_check_adx_accel(self, pos, bar, bar_index, inst) -> bool:
        if self.check_bar is None or self.adx_accel_threshold is None:
            return False
        key = id(pos)
        hist = self._adx_history.setdefault(key, [])
        hist.append(bar["adx"])

        bars_held = bar_index - pos.entry_bar_index
        if bars_held != self.check_bar:
            return False
        if len(hist) < 3 or pd.isna(hist[-1]) or pd.isna(hist[-2]) or pd.isna(hist[-3]):
            return False

        a2, a1, a0 = hist[-3], hist[-2], hist[-1]
        adx_accel = (a0 - a1) - (a1 - a2)
        if adx_accel < self.adx_accel_threshold:
            spread = inst.spread_fixed
            exit_price = (bar["close"] - spread / 2 if pos.direction == "long"
                          else bar["close"] + spread / 2)
            self._close_position(pos, bar["timestamp"], exit_price, "adx_decel_exit")
            self._adx_history.pop(key, None)
            return True
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

                closed = self._try_close_position(pos, bar, bar_index, inst)
                if closed:
                    self._adx_history.pop(id(pos), None)
                    continue

                # --- NUOVO: aggiorna storia ADX e controlla decelerazione alla barra 4 ---
                self._update_and_check_adx_accel(pos, bar, bar_index, inst)
                # --- fine nuovo ---

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


# =====================================================================
# 2) BOOTSTRAP A RISCHIO FISSO
# =====================================================================

def build_day_index(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    df = trades_df.copy()
    df["entry_day"] = pd.to_datetime(df["entry_time"]).dt.date
    df["risk_pct_used"] = df["instrument"].map(
        {name: inst.risk_pct for name, inst in eng.INSTRUMENTS.items()})
    df = df.sort_values("entry_time")
    out = {}
    for day, grp in df.groupby("entry_day"):
        out[day] = list(zip(grp["risk_pct_used"], grp["r_multiple"]))
    return out


def replay_path_fixed_risk(day_sequence: list, day_index: dict, capital0: float) -> float:
    total_pnl = 0.0
    for day in day_sequence:
        trades_today = day_index.get(day, [])
        if not trades_today:
            continue
        day_pnl = 0.0
        kill_switch_active = False
        for risk_pct, r_mult in trades_today:
            if kill_switch_active:
                continue
            risk_amount = capital0 * risk_pct
            pnl = r_mult * risk_amount
            day_pnl += pnl
            if (day_pnl / capital0) <= -KILL_SWITCH_PCT:
                kill_switch_active = True
        total_pnl += day_pnl
    return total_pnl


def bootstrap_period(base_trades: pd.DataFrame, variant_trades: pd.DataFrame,
                      all_days: list, capital0: float, n_boot: int,
                      rng: np.random.Generator) -> dict:
    base_idx = build_day_index(base_trades)
    variant_idx = build_day_index(variant_trades)
    n_days = len(all_days)

    observed_base = replay_path_fixed_risk(all_days, base_idx, capital0)
    observed_variant = replay_path_fixed_risk(all_days, variant_idx, capital0)
    observed_delta = observed_variant - observed_base

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = [all_days[i] for i in rng.integers(0, n_days, size=n_days)]
        pnl_base = replay_path_fixed_risk(sampled_days, base_idx, capital0)
        pnl_variant = replay_path_fixed_risk(sampled_days, variant_idx, capital0)
        deltas[b] = pnl_variant - pnl_base

    return {
        "observed_delta": observed_delta, "boot_mean": deltas.mean(), "boot_sd": deltas.std(),
        "ci_low_95": np.percentile(deltas, 2.5), "ci_high_95": np.percentile(deltas, 97.5),
        "pct_iter_non_positive": (deltas <= 0).mean() * 100, "deltas": deltas,
    }


# =====================================================================
# 3) MAIN — sanity check (una volta) -> entrambe le varianti fino in fondo -> confronto
# =====================================================================

def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def run_variant(label, check_bar, threshold, signals_full, rng):
    rows = []
    total_deltas = np.zeros(N_BOOT)
    total_observed_delta = 0.0
    total_trades_base = 0
    total_trades_variant = 0
    total_decel_exits = 0

    for period_label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
        trades_base, metrics_base = eng_base.run(sig_period)

        eng_variant = BacktestEngineADXAccelExit(
            capital0=CAPITAL0, instruments=eng.INSTRUMENTS,
            check_bar=check_bar, adx_accel_threshold=threshold)
        trades_variant, metrics_variant = eng_variant.run(sig_period)

        n_decel = (trades_variant["exit_reason"] == "adx_decel_exit").sum() if not trades_variant.empty else 0
        total_decel_exits += n_decel

        mb, mv = metrics_base.iloc[0], metrics_variant.iloc[0]
        print(f"  [{label}] {period_label}: baseline={len(trades_base)} trade pnl={mb['pnl_total']:+.2f}  |  "
              f"variante={len(trades_variant)} trade ({n_decel} usciti per decelerazione ADX) pnl={mv['pnl_total']:+.2f}")

        all_days = sorted(set(sig_period["DAX"]["timestamp"].dt.date) |
                           set(sig_period["FTSE100"]["timestamp"].dt.date))
        result = bootstrap_period(trades_base, trades_variant, all_days, CAPITAL0, N_BOOT, rng)
        print(f"      bootstrap: delta={result['observed_delta']:+.2f}  sd={result['boot_sd']:.2f}  "
              f"IC95%=[{result['ci_low_95']:+.2f}, {result['ci_high_95']:+.2f}]  "
              f"%iter<=0={result['pct_iter_non_positive']:.1f}%")

        rows.append({
            "variante": label, "periodo": period_label,
            "n_trade_base": len(trades_base), "n_trade_variant": len(trades_variant),
            "n_decel_exits": int(n_decel),
            "pnl_base": mb["pnl_total"], "pnl_variant": mv["pnl_total"],
            "delta_osservato": result["observed_delta"], "boot_sd": result["boot_sd"],
            "ci95_low": result["ci_low_95"], "ci95_high": result["ci_high_95"],
            "pct_iter_delta_non_positivo": result["pct_iter_non_positive"],
        })
        total_deltas += result["deltas"]
        total_observed_delta += result["observed_delta"]
        total_trades_base += len(trades_base)
        total_trades_variant += len(trades_variant)

    z_tot = total_observed_delta / total_deltas.std() if total_deltas.std() > 0 else float("nan")
    ci_low_tot, ci_high_tot = np.percentile(total_deltas, [2.5, 97.5])
    pct_non_positive_tot = (total_deltas <= 0).mean() * 100

    summary = {
        "variante": label, "check_bar": check_bar, "threshold": threshold,
        "trade_totali_base": total_trades_base, "trade_totali_variant": total_trades_variant,
        "decel_exits_totali": total_decel_exits,
        "delta_osservato_totale": total_observed_delta,
        "boot_sd_totale": total_deltas.std(), "ci95_low_totale": ci_low_tot, "ci95_high_totale": ci_high_tot,
        "pct_iter_non_positivo_totale": pct_non_positive_tot, "z_approx": z_tot,
    }
    return rows, summary


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    print("Scarico/aggiorno storico DAX/FTSE100...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in SYMBOLS}
    signals_full = {name: eng.generate_signals(raw[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    print("Fatto.\n")

    # --- SANITY CHECK (una volta sola) ---
    print("=== SANITY CHECK (adx_accel_threshold=None deve dare risultati identici al motore standard) ===")
    p_start, p_end = pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    sig_check = {name: slice_period(signals_full[name], p_start, p_end) for name in SYMBOLS}

    baseline_check = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
    trades_bc, _ = baseline_check.run(sig_check)
    variant_check = BacktestEngineADXAccelExit(
        capital0=CAPITAL0, instruments=eng.INSTRUMENTS, check_bar=None, adx_accel_threshold=None)
    trades_vc, _ = variant_check.run(sig_check)

    same_n = len(trades_bc) == len(trades_vc)
    same_capital = abs(baseline_check.capital - variant_check.capital) < 0.01
    same_pnl = same_n and np.allclose(trades_bc["pnl"].values, trades_vc["pnl"].values, atol=1e-6)
    if same_n and same_capital and same_pnl:
        print(f"SANITY CHECK: PASS ({len(trades_bc)} trade identici, capitale identico)\n")
    else:
        print(f"SANITY CHECK: FALLITO — baseline={len(trades_bc)} trade/{baseline_check.capital:.2f}, "
              f"variante={len(trades_vc)} trade/{variant_check.capital:.2f}. STOP, non procedo oltre.")
        return

    # --- ENTRAMBE LE VARIANTI, fino in fondo ---
    rng = np.random.default_rng(SEED)
    all_rows = []
    all_summaries = []
    for label, check_bar, threshold in VARIANTS:
        print(f"=== Variante {label}: check_bar={check_bar}, adx_accel_threshold={threshold} ===")
        rows, summary = run_variant(label, check_bar, threshold, signals_full, rng)
        all_rows.extend(rows)
        all_summaries.append(summary)
        print()

    print(f"{'='*70}\nCONFRONTO FINALE — entrambe le varianti, aggregato sui 5 periodi\n{'='*70}")
    for s in all_summaries:
        print(f"\n{s['variante']} (check_bar={s['check_bar']}, adx_accel_threshold={s['threshold']}):")
        print(f"  trade totali: baseline={s['trade_totali_base']}  variante={s['trade_totali_variant']}  "
              f"(di cui {s['decel_exits_totali']} usciti per decelerazione ADX)")
        print(f"  delta osservato totale: {s['delta_osservato_totale']:+.2f} EUR")
        print(f"  IC 95% del rumore atteso: [{s['ci95_low_totale']:+.2f}, {s['ci95_high_totale']:+.2f}]")
        print(f"  % iterazioni bootstrap con delta<=0: {s['pct_iter_non_positivo_totale']:.1f}%")
        print(f"  z-score approssimato: {s['z_approx']:.2f}")

    print(f"\nVERDETTO: se il delta osservato cade dentro l'IC 95% del rumore, o se >10-15% delle "
          f"iterazioni danno delta<=0, la variante NON e' chiaramente distinguibile dal rumore "
          f"campionario (Protocollo Anti-Rumore, principio 4).")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(all_rows).to_csv("results/test_adx_accel_exit_dettaglio.csv", index=False)
    pd.DataFrame(all_summaries).to_csv("results/test_adx_accel_exit_riepilogo.csv", index=False)
    print("\nDettaglio salvato in results/test_adx_accel_exit_dettaglio.csv "
          "e results/test_adx_accel_exit_riepilogo.csv")


if __name__ == "__main__":
    main()
