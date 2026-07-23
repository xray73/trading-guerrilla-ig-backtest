"""
test_calibration_controls_v2.py — Versione corretta dopo che ENTRAMBI
i controlli v1 sono risultati fuori attesa per DIFETTI DI DESIGN del
controllo stesso, non del bootstrap:

  - v1 controllo negativo (uscita a moneta): NON era neutro — un'uscita
    casuale ha piu' occasioni di troncare un vincente (che impiega piu'
    barre per arrivare al target 2R) che un perdente (stop piu' vicino,
    1R) PRIMA che maturi. Stesso bias strutturale scoperto oggi con
    idea 1, non un difetto del bootstrap.
  - v1 controllo positivo (filtro ADX): confondeva qualita' con VOLUME
    di trade — rimuovere il filtro fa passare piu' segnali (942 vs 750
    in un periodo), e con size legata al capitale corrente piu' trade
    presto possono gonfiare il PnL assoluto anche con qualita' media
    peggiore per trade. Il filtro ADX non e' mai stato validato per
    "massimizza il PnL testa a testa senza filtro", quindi il confronto
    misurava la variabile sbagliata.

CORREZIONI v2:
  - CONTROLLO NEGATIVO: selezione casuale ALL'INGRESSO (coin-flip su
    quali segnali candidati si accettano, prob=0.5), confrontando DUE
    seed diversi (nessuno dei due e' "il baseline vero") — tocca SOLO
    quali trade si aprono, MAI la durata/uscita, quindi l'asimmetria
    R:R strutturale non entra in gioco. Atteso: z vicino a 0 tra le due
    selezioni casuali indipendenti.
  - CONTROLLO POSITIVO: spread raddoppiato vs spread standard, STESSI
    segnali, STESSO numero di trade esatto (lo spread influenza solo
    prezzo di entrata/uscita, mai se un segnale scatta) — isola
    l'effetto costo di esecuzione, gia' documentato nel progetto come
    reale (RCA: spread realistico riduce PnL ~36%). Atteso: z ben
    oltre 2.0 a favore dello spread standard.

Nessuna scrittura su D1.
"""
import os
import sys
import copy
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

RANDOM_FILTER_PROB = 0.5
SPREAD_MULTIPLIER = 2.0


class BacktestEngineRandomEntryFilter(BacktestEngineFloatingKillSwitch):
    def __init__(self, capital0, seed: int, keep_prob: float = RANDOM_FILTER_PROB, **kwargs):
        super().__init__(capital0, **kwargs)
        self._rng = np.random.default_rng(seed)
        self.keep_prob = keep_prob

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

                if self._rng.random() >= self.keep_prob:
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


def make_high_spread_instruments(multiplier: float) -> dict:
    out = {}
    for name, inst in eng.INSTRUMENTS.items():
        new_inst = copy.deepcopy(inst)
        new_inst.spread_fixed = inst.spread_fixed * multiplier
        out[name] = new_inst
    return out


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} PERIODO_HOLDOUT")
        sys.exit(1)
    holdout_label = sys.argv[1].strip()
    if holdout_label not in PERIODS:
        print(f"ERRORE: periodo '{holdout_label}' non riconosciuto.")
        sys.exit(1)

    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6 standard...")
    signals_standard = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    high_spread_instruments = make_high_spread_instruments(SPREAD_MULTIPLIER)

    def random_filter_factory_seedA(cap):
        return BacktestEngineRandomEntryFilter(capital0=cap, seed=111)

    def random_filter_factory_seedB(cap):
        return BacktestEngineRandomEntryFilter(capital0=cap, seed=222)

    def std_spread_factory(cap):
        return BacktestEngineFloatingKillSwitch(capital0=cap)

    def high_spread_factory(cap):
        return BacktestEngineFloatingKillSwitch(capital0=cap, instruments=high_spread_instruments)

    print("\n" + "#" * 90)
    print("# CONTROLLO NEGATIVO v2 — selezione casuale ingresso, seed A vs seed B")
    print("#" * 90)

    print("=== SANITY CHECK (determinismo, non riproduzione standard) ===")
    start, end = PERIODS["2015-2016"]
    t1, _ = cf.run_period(random_filter_factory_seedA, signals_standard, start, end, CAPITAL_V6)
    t2, _ = cf.run_period(random_filter_factory_seedA, signals_standard, start, end, CAPITAL_V6)
    if len(t1) != len(t2) or abs(float(t1["pnl"].sum()) - float(t2["pnl"].sum())) > 0.01:
        print("  *** FALLITO: stesso seed produce risultati diversi tra run. Interrompo.")
        sys.exit(1)
    print(f"  Seed 111 rieseguito due volte: {len(t1)} trade entrambe le volte, "
          f"PnL identico {float(t1['pnl'].sum()):+.2f} EUR. OK\n")

    res_neg_holdout = cf.bootstrap_compare(
        random_filter_factory_seedA, signals_standard, random_filter_factory_seedB, signals_standard,
        [holdout_label], PERIODS, CAPITAL_V6)
    cf.print_result(f"CONTROLLO NEGATIVO v2 — solo holdout ({holdout_label})", res_neg_holdout)

    res_neg_all = cf.bootstrap_compare(
        random_filter_factory_seedA, signals_standard, random_filter_factory_seedB, signals_standard,
        list(PERIODS.keys()), PERIODS, CAPITAL_V6)
    cf.print_result("CONTROLLO NEGATIVO v2 — tutti i 5 periodi", res_neg_all)

    print(f"\n>>> Atteso: |z| vicino a 0. Osservato: holdout z={res_neg_holdout['z_score']:.3f}, "
          f"aggregato z={res_neg_all['z_score']:.3f}")

    print("\n" + "#" * 90)
    print("# CONTROLLO POSITIVO v2 — spread raddoppiato vs standard, stessi trade")
    print("#" * 90)

    cf.sanity_check(std_spread_factory, signals_standard,
                     std_spread_factory, signals_standard,
                     PERIODS["2015-2016"], CAPITAL_V6, label="(controllo positivo v2, banale)")

    res_pos_holdout = cf.bootstrap_compare(
        high_spread_factory, signals_standard, std_spread_factory, signals_standard,
        [holdout_label], PERIODS, CAPITAL_V6)
    cf.print_result(f"CONTROLLO POSITIVO v2 — solo holdout ({holdout_label})", res_pos_holdout)

    res_pos_all = cf.bootstrap_compare(
        high_spread_factory, signals_standard, std_spread_factory, signals_standard,
        list(PERIODS.keys()), PERIODS, CAPITAL_V6)
    cf.print_result("CONTROLLO POSITIVO v2 — tutti i 5 periodi", res_pos_all)

    print(f"\n>>> Atteso: z ben oltre 2.0 (spread standard batte spread raddoppiato, "
          f"stesso numero di trade). Osservato: holdout z={res_pos_holdout['z_score']:.3f}, "
          f"aggregato z={res_pos_all['z_score']:.3f}")

    print("\n" + "=" * 90)
    print("VERDETTO CALIBRAZIONE STRUMENTI (v2)")
    print("=" * 90)
    neg_ok = abs(res_neg_all["z_score"]) < 2.0
    pos_ok = res_pos_all["z_score"] >= 2.0
    print(f"Controllo negativo v2 (atteso ~0): z aggregato={res_neg_all['z_score']:.3f} -> entro attese: {neg_ok}")
    print(f"Controllo positivo v2 (atteso >>2): z aggregato={res_pos_all['z_score']:.3f} -> entro attese: {pos_ok}")
    if neg_ok and pos_ok:
        print("\nIl bootstrap risulta ben calibrato su entrambi i controlli corretti: "
              "non genera falsi positivi su selezione casuale pura, e riconosce con "
              "margine netto un effetto vero e noto (costo di esecuzione).")
    else:
        print("\nAlmeno un controllo resta fuori attesa anche nella versione corretta — "
              "merita un'indagine dedicata separata prima di continuare a fidarsi ciecamente.")


if __name__ == "__main__":
    main()
