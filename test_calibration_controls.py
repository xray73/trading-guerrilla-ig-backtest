"""
test_calibration_controls.py — Verifica la CALIBRAZIONE degli
strumenti di validazione, non un'ipotesi di trading. Due controlli
indipendenti nello stesso run (nessuno stato condiviso tra i due,
sicuro farli insieme):

CONTROLLO NEGATIVO (falso positivo atteso raro): un meccanismo che
chiude posizioni in base a un lancio di moneta (probabilita' fissa
per barra, seed fisso, scollegato da qualunque segnale reale) NON
dovrebbe mostrare un effetto — z dovrebbe restare vicino a 0, superare
2.0 solo raramente. Se lo superasse spesso, il bootstrap sottostima il
rumore e gli z gia' misurati oggi sono meno affidabili di quanto
creduto.

CONTROLLO POSITIVO (vero effetto atteso, gia' noto): il filtro ADX>20
dentro il segnale V6 e' parte del sistema validato in produzione — la
sua rimozione dovrebbe mostrare un effetto NETTAMENTE negativo (cioe'
il sistema CON filtro batte nettamente il sistema SENZA filtro), con z
ben oltre 2.0. Se anche questo faticasse a superare la soglia, il
bootstrap sarebbe troppo conservativo per questo tipo di campioni.

Entrambi usano lo stesso identico protocollo (sanity check, bootstrap
a blocchi di giornata N=2000, seed fisso) tramite causal_framework.py
— cosi' il confronto tra "quanto e' forte un vero effetto" e "quanto
rumore genera un meccanismo senza effetto" e' diretto.

Nessuna scrittura su D1.
"""
import os
import sys
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc
import causal_framework as cf

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

CAPITAL_V6 = 1400.0

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}

COIN_FLIP_PROB = 0.05  # probabilita' per barra di chiusura casuale, controllo negativo


class BacktestEngineCoinFlipExit(BacktestEngineFloatingKillSwitch):
    def __init__(self, capital0, flip_prob: float = COIN_FLIP_PROB, seed: int = 123, **kwargs):
        super().__init__(capital0, **kwargs)
        self.flip_prob = flip_prob
        self._rng = np.random.default_rng(seed)

    def run(self, data):
        tradable_instruments = [
            name for name in data
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")
        all_timestamps = sorted(set().union(*[set(data[i]["timestamp"]) for i in tradable_instruments]))

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

                if self.flip_prob > 0 and self._rng.random() < self.flip_prob:
                    spread = inst.spread_fixed
                    exit_price = (bar["close"] - spread / 2 if pos.direction == "long"
                                  else bar["close"] + spread / 2)
                    self._close_position(pos, bar["timestamp"], exit_price, "coin_flip_exit")
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
                self._open_position(c["instrument"], c["direction"], c["bar"], c["atr"], c["adx"])
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


def generate_signals_no_adx_filter(df: pd.DataFrame, inst: eng.InstrumentConfig,
                                    p: eng.ChartaParams = eng.PARAMS) -> pd.DataFrame:
    out = eng.compute_indicators(df, inst, p)

    direction_long = out["ema_fast"] > out["ema_slow"]
    direction_short = out["ema_fast"] < out["ema_slow"]

    breakout_long = out["close"] > out["rolling_high"]
    breakout_short = out["close"] < out["rolling_low"]

    broad_trend_long_ok = out["ema_broad_fast"] > out["ema_broad_slow"]
    broad_trend_short_ok = out["ema_broad_fast"] < out["ema_broad_slow"]

    long_signal = direction_long & breakout_long & broad_trend_long_ok
    short_signal = direction_short & breakout_short & broad_trend_short_ok

    out["signal"] = None
    out.loc[long_signal, "signal"] = "long"
    out.loc[short_signal, "signal"] = "short"
    return out


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} PERIODO_HOLDOUT")
        print(f"Periodi disponibili: {', '.join(PERIODS)}")
        sys.exit(1)
    holdout_label = sys.argv[1].strip()
    if holdout_label not in PERIODS:
        print(f"ERRORE: periodo '{holdout_label}' non riconosciuto.")
        sys.exit(1)

    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6 standard (con filtro ADX)...")
    signals_standard = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Genero segnali V6 SENZA filtro ADX (per controllo positivo)...")
    signals_no_adx = {name: generate_signals_no_adx_filter(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    def std_engine_factory(cap):
        return BacktestEngineFloatingKillSwitch(capital0=cap)

    def coinflip_engine_factory(cap):
        return BacktestEngineCoinFlipExit(capital0=cap, flip_prob=COIN_FLIP_PROB)

    def coinflip_neutral_factory(cap):
        return BacktestEngineCoinFlipExit(capital0=cap, flip_prob=0.0)

    print("\n" + "#" * 90)
    print("# CONTROLLO NEGATIVO — uscita a moneta scollegata dal segnale")
    print("#" * 90)

    ok = cf.sanity_check(std_engine_factory, signals_standard,
                          coinflip_neutral_factory, signals_standard,
                          PERIODS["2015-2016"], CAPITAL_V6, label="(controllo negativo)")
    if not ok:
        print("Sanity check controllo negativo fallito, interrompo.")
        sys.exit(1)

    res_neg_holdout = cf.bootstrap_compare(
        std_engine_factory, signals_standard, coinflip_engine_factory, signals_standard,
        [holdout_label], PERIODS, CAPITAL_V6)
    cf.print_result(f"CONTROLLO NEGATIVO — solo holdout ({holdout_label})", res_neg_holdout)

    res_neg_all = cf.bootstrap_compare(
        std_engine_factory, signals_standard, coinflip_engine_factory, signals_standard,
        list(PERIODS.keys()), PERIODS, CAPITAL_V6)
    cf.print_result("CONTROLLO NEGATIVO — tutti i 5 periodi", res_neg_all)

    print(f"\n>>> Atteso: |z| vicino a 0, NON dovrebbe superare 2.0. "
          f"Osservato: holdout z={res_neg_holdout['z_score']:.3f}, aggregato z={res_neg_all['z_score']:.3f}")

    print("\n" + "#" * 90)
    print("# CONTROLLO POSITIVO — filtro ADX>20 presente vs assente")
    print("#" * 90)

    cf.sanity_check(std_engine_factory, signals_standard,
                     std_engine_factory, signals_standard,
                     PERIODS["2015-2016"], CAPITAL_V6, label="(controllo positivo, banale)")

    res_pos_holdout = cf.bootstrap_compare(
        std_engine_factory, signals_no_adx, std_engine_factory, signals_standard,
        [holdout_label], PERIODS, CAPITAL_V6)
    cf.print_result(f"CONTROLLO POSITIVO — solo holdout ({holdout_label})", res_pos_holdout)

    res_pos_all = cf.bootstrap_compare(
        std_engine_factory, signals_no_adx, std_engine_factory, signals_standard,
        list(PERIODS.keys()), PERIODS, CAPITAL_V6)
    cf.print_result("CONTROLLO POSITIVO — tutti i 5 periodi", res_pos_all)

    print(f"\n>>> Atteso: z ben oltre 2.0 (il filtro ADX e' un effetto reale gia' validato). "
          f"Osservato: holdout z={res_pos_holdout['z_score']:.3f}, aggregato z={res_pos_all['z_score']:.3f}")

    print("\n" + "=" * 90)
    print("VERDETTO CALIBRAZIONE STRUMENTI")
    print("=" * 90)
    print(f"Controllo negativo (atteso ~0): z holdout={res_neg_holdout['z_score']:.3f}, "
          f"z aggregato={res_neg_all['z_score']:.3f}")
    print(f"Controllo positivo (atteso >>2): z holdout={res_pos_holdout['z_score']:.3f}, "
          f"z aggregato={res_pos_all['z_score']:.3f}")
    neg_ok = abs(res_neg_all["z_score"]) < 2.0
    pos_ok = res_pos_all["z_score"] >= 2.0
    print(f"\nControllo negativo entro attese: {neg_ok}")
    print(f"Controllo positivo entro attese: {pos_ok}")
    if neg_ok and pos_ok:
        print("Il protocollo di validazione appare ben calibrato: non genera falsi "
              "positivi su rumore puro, e riconosce un effetto vero gia' noto.")
    else:
        print("ATTENZIONE: almeno un controllo e' fuori dalle attese — il protocollo "
              "potrebbe necessitare revisione prima di fidarsi ciecamente degli z "
              "misurati finora.")


if __name__ == "__main__":
    main()
