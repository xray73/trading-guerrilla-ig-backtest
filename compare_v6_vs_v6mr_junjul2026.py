"""
compare_v6_vs_v6mr_junjul2026.py — Confronto ad-hoc (18/07/2026):
"vecchio motore" (V6 pura, 2.000€ interi, con accantonamento) contro
"nuovo motore" (V6 70% + mean-reversion RSI 30%, split fisso su 2.000€,
accantonamento combinato mensile, kill switch separato per pool) sul
periodo 01/06/2026 - 18/07/2026 (dati mai visti nei backtest ufficiali,
che si fermano ai 5 periodi storici).

SPREAD: usa spread_fixed originale di engine.py (DAX 1,2pt, FTSE100
1,0pt) — quello con cui il motore è specificato oggi, NON il valore
realistico scoperto il 17/07 (DAX ~2,5-2,7pt, FTSE100 ~1,5-1,7pt).
Risultati quindi ottimistici rispetto a IG reale, stessa base di
confronto di tutti i backtest ufficiali del progetto (nessuna
distorsione relativa tra i due scenari, ma entrambi da leggere con lo
stesso caveat già noto: PnL totale reale atteso ~-36% più basso,
vedi spread_sensitivity_revalidation.py).

METODOLOGIA:
  - Dati: dukascopy_python diretto (non D1, che si ferma al 10/07),
    storico da 2025-01-01 per warmup EMA200/RSI/Bollinger sufficiente,
    fino a oggi.
  - Scenario A ("vecchio motore"): BacktestEngineAccantonamento
    (eredita BacktestEngineFloatingKillSwitch), capitale 2.000€ interi,
    segnale V6 invariato, accantonamento "opzione 3 mensile" (soglia
    1.5x, consolida 40%, già default della classe). UN SOLO run
    continuo sul periodo intero.
  - Scenario B ("nuovo motore"): due motori SEPARATI (V6 1.400€ via
    BacktestEngineFloatingKillSwitch, mean-reversion RSI 600€ via
    BacktestEngineMeanReversion) — replica esatta della logica ora in
    live_execute.py. Kill switch già separato per pool per costruzione
    (ogni istanza controlla solo il proprio self.capital). Poiché
    l'accantonamento qui è COMBINATO (decisione 18/07/2026) e i due
    pool devono essere ridotti insieme quando scatta, i motori sono
    eseguiti IN BLOCCHI MENSILI: fine mese, si sommano i due capitali,
    si applica la stessa formula di consolidamento di
    apply_monthly_consolidation_if_needed() in live_execute.py, si
    riduce ciascun pool in proporzione, si passa il capitale
    aggiornato al blocco mensile successivo.

SEMPLIFICAZIONE NOTA (accettabile per un confronto ad-hoc, non per una
validazione formale): ogni blocco mensile usa un motore FRESCO (nuova
istanza), quindi una posizione ancora aperta esattamente a cavallo del
cambio mese non prosegue nel blocco successivo. Impatto atteso
trascurabile (max_holding=24h, un solo confine di mese nel periodo
testato, quindi al più 1-2 trade coinvolti).

Metriche riportate con la stessa metodologia in entrambi gli scenari
(eng.compute_run_metrics su trade combinati, capitale iniziale 2.000€
come base per il drawdown %) per comparabilità diretta.

Nessuna scrittura su D1. Solo stampa a log + file risultati/ per l'artifact.
"""

from __future__ import annotations

import os
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100
from datetime import datetime, timezone

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_accantonamento import BacktestEngineAccantonamento
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals

SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}
WARMUP_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_START = datetime(2026, 6, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 7, 19, tzinfo=timezone.utc)  # esclusivo, copre fino al 18/07 incluso

CAPITAL0 = 2000.0
SPLIT_V6, SPLIT_MR = 0.70, 0.30
MR_MODE = "rsi"

MONTH_CHUNKS = [
    (datetime(2026, 6, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc)),
    (datetime(2026, 7, 1, tzinfo=timezone.utc), TEST_END),
]


def fetch_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        WARMUP_START, TEST_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_df(df: pd.DataFrame, start: datetime, end: datetime, buffer_days: int = 2) -> pd.DataFrame:
    buf_start = pd.Timestamp(start) - pd.Timedelta(days=buffer_days)
    out = df[(df["timestamp"] >= buf_start) & (df["timestamp"] < pd.Timestamp(end))].reset_index(drop=True)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Confronto V6 pura vs V6+MR — periodo {TEST_START.date()} -> {(TEST_END - pd.Timedelta(days=1)).date()} ===")
    log(f"Spread: originale engine.py (DAX={eng.INSTRUMENTS['DAX'].spread_fixed}pt, "
        f"FTSE100={eng.INSTRUMENTS['FTSE100'].spread_fixed}pt) — NON il valore realistico scoperto il 17/07.\n")

    log("Scarico storico DAX/FTSE100 (dukascopy diretto, non D1)...")
    hist_full = {}
    for name, const in SYMBOLS.items():
        hist_full[name] = fetch_full(const)
        log(f"  {name}: {len(hist_full[name])} barre, {hist_full[name]['timestamp'].min()} -> {hist_full[name]['timestamp'].max()}")

    v6_signals_full = {name: eng.generate_signals(hist_full[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    mr_signals_full = {name: generate_mean_reversion_signals(hist_full[name], eng.INSTRUMENTS[name], mode=MR_MODE) for name in SYMBOLS}

    # ================================================================
    # SCENARIO A — vecchio motore (V6 pura, 2000 EUR, accantonamento)
    # ================================================================
    log("\n" + "=" * 70)
    log("SCENARIO A — Vecchio motore: V6 pura, 2.000 EUR, con accantonamento")
    log("=" * 70)

    v6_sliced_a = {name: slice_df(v6_signals_full[name], TEST_START, TEST_END) for name in SYMBOLS}
    engine_a = BacktestEngineAccantonamento(capital0=CAPITAL0, mode="gradini", check_frequency="mensile")
    trades_a, metrics_a = engine_a.run(v6_sliced_a)

    m = metrics_a.iloc[0]
    log(f"  Trade totali: {int(m['num_trades'])}")
    log(f"  Win rate: {m['win_rate']*100:.1f}%" if pd.notna(m['win_rate']) else "  Win rate: N/A")
    log(f"  Profit Factor: {m['profit_factor']:.2f}" if pd.notna(m['profit_factor']) else "  PF: N/A")
    log(f"  PnL totale (trading, ante accantonamento): {m['pnl_total']:+.2f} EUR")
    log(f"  Max drawdown: {m['max_drawdown_pct']*100:.2f}%" if pd.notna(m['max_drawdown_pct']) else "  MaxDD: N/A")
    log(f"  Capitale investito finale: {m['capitale_investito_finale']:.2f} EUR")
    log(f"  Accantonato finale: {m['accantonato_finale']:.2f} EUR")
    log(f"  Capitale TOTALE finale (investito+accantonato): {m['capitale_totale_finale']:.2f} EUR")
    log(f"  Rendimento su 2.000 EUR: {(m['capitale_totale_finale']/CAPITAL0 - 1)*100:+.2f}%")
    log(f"  Numero consolidamenti scattati: {int(m['n_consolidamenti'])}")

    # ================================================================
    # SCENARIO B — nuovo motore (V6 70% + MR 30%, accantonamento combinato)
    # ================================================================
    log("\n" + "=" * 70)
    log("SCENARIO B — Nuovo motore: V6 70% + mean-reversion RSI 30%, "
        "accantonamento combinato mensile, kill switch separato per pool")
    log("=" * 70)

    cap_v6 = CAPITAL0 * SPLIT_V6
    cap_mr = CAPITAL0 * SPLIT_MR
    reference_combined = CAPITAL0
    threshold_combined = reference_combined * 1.5
    accantonato_combined = 0.0
    n_consolidamenti_b = 0

    all_trades_b = []

    for i, (chunk_start, chunk_end) in enumerate(MONTH_CHUNKS):
        label = f"{chunk_start.date()} -> {(chunk_end - pd.Timedelta(days=1)).date()}"
        log(f"\n  --- Blocco mensile {i+1}: {label} ---")

        v6_sliced = {name: slice_df(v6_signals_full[name], chunk_start, chunk_end) for name in SYMBOLS}
        mr_sliced = {name: slice_df(mr_signals_full[name], chunk_start, chunk_end) for name in SYMBOLS}

        eng_v6 = BacktestEngineFloatingKillSwitch(capital0=cap_v6)
        trades_v6, _ = eng_v6.run(v6_sliced)
        cap_v6_end = eng_v6.capital

        eng_mr = BacktestEngineMeanReversion(capital0=cap_mr)
        trades_mr, _ = eng_mr.run(mr_sliced)
        cap_mr_end = eng_mr.capital

        if not trades_v6.empty:
            trades_v6 = trades_v6.copy(); trades_v6["strategy"] = "v6"
            all_trades_b.append(trades_v6)
        if not trades_mr.empty:
            trades_mr = trades_mr.copy(); trades_mr["strategy"] = "mean_reversion"
            all_trades_b.append(trades_mr)

        log(f"    V6: {len(trades_v6)} trade, capitale pool {cap_v6:.2f} -> {cap_v6_end:.2f} EUR "
            f"(kill switch attivato: {eng_v6._kill_switch_active})")
        log(f"    MR: {len(trades_mr)} trade, {eng_mr.n_skipped_min_size} saltati per size, "
            f"capitale pool {cap_mr:.2f} -> {cap_mr_end:.2f} EUR "
            f"(kill switch attivato: {eng_mr._kill_switch_active})")

        combined = cap_v6_end + cap_mr_end
        while combined > threshold_combined:
            gain = combined - reference_combined
            consolidated = 0.4 * gain
            if consolidated <= 0:
                break
            frac = consolidated / combined
            cap_v6_end -= cap_v6_end * frac
            cap_mr_end -= cap_mr_end * frac
            accantonato_combined += consolidated
            n_consolidamenti_b += 1
            combined = cap_v6_end + cap_mr_end
            reference_combined = combined
            threshold_combined = reference_combined * 1.5
            log(f"    [accantonamento] Consolidati {consolidated:.2f} EUR — "
                f"investito totale ora {combined:.2f} EUR, accantonato cumulato {accantonato_combined:.2f} EUR")

        cap_v6, cap_mr = cap_v6_end, cap_mr_end

    combined_trades_b = pd.concat(all_trades_b, ignore_index=True) if all_trades_b else pd.DataFrame()
    combined_trades_b = combined_trades_b.sort_values("entry_time").reset_index(drop=True) if not combined_trades_b.empty else combined_trades_b

    metrics_b = eng.compute_run_metrics(combined_trades_b, CAPITAL0, cap_v6 + cap_mr)
    mb = metrics_b.iloc[0]

    log(f"\n  --- Riepilogo Scenario B ---")
    log(f"  Trade totali (V6+MR): {int(mb['num_trades'])}")
    n_v6 = int((combined_trades_b['strategy'] == 'v6').sum()) if not combined_trades_b.empty else 0
    n_mr = int((combined_trades_b['strategy'] == 'mean_reversion').sum()) if not combined_trades_b.empty else 0
    log(f"    di cui V6: {n_v6}, mean-reversion: {n_mr}")
    log(f"  Win rate: {mb['win_rate']*100:.1f}%" if pd.notna(mb['win_rate']) else "  Win rate: N/A")
    log(f"  Profit Factor: {mb['profit_factor']:.2f}" if pd.notna(mb['profit_factor']) else "  PF: N/A")
    log(f"  PnL totale (trading, ante accantonamento): {mb['pnl_total']:+.2f} EUR")
    log(f"  Max drawdown: {mb['max_drawdown_pct']*100:.2f}%" if pd.notna(mb['max_drawdown_pct']) else "  MaxDD: N/A")
    log(f"  Capitale investito finale: {cap_v6 + cap_mr:.2f} EUR (V6={cap_v6:.2f} MR={cap_mr:.2f})")
    log(f"  Accantonato finale: {accantonato_combined:.2f} EUR")
    log(f"  Capitale TOTALE finale: {cap_v6 + cap_mr + accantonato_combined:.2f} EUR")
    log(f"  Rendimento su 2.000 EUR: {((cap_v6 + cap_mr + accantonato_combined)/CAPITAL0 - 1)*100:+.2f}%")
    log(f"  Numero consolidamenti scattati: {n_consolidamenti_b}")

    # ================================================================
    # CONFRONTO DIRETTO
    # ================================================================
    log("\n" + "=" * 70)
    log("CONFRONTO DIRETTO")
    log("=" * 70)
    totale_a = m['capitale_totale_finale']
    totale_b = cap_v6 + cap_mr + accantonato_combined
    log(f"  Capitale totale finale — A (V6 pura): {totale_a:.2f} EUR  |  B (V6+MR): {totale_b:.2f} EUR")
    log(f"  Differenza: {totale_b - totale_a:+.2f} EUR ({(totale_b/totale_a - 1)*100:+.2f}% relativo a A)")
    dd_a = m['max_drawdown_pct']*100 if pd.notna(m['max_drawdown_pct']) else float('nan')
    dd_b = mb['max_drawdown_pct']*100 if pd.notna(mb['max_drawdown_pct']) else float('nan')
    log(f"  Max drawdown — A: {dd_a:.2f}%  |  B: {dd_b:.2f}%")
    log(f"  Accantonato — A: {m['accantonato_finale']:.2f} EUR  |  B: {accantonato_combined:.2f} EUR")

    with open("results/compare_v6_vs_v6mr_junjul2026.txt", "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
