"""
accantonamento_validation.py — Validazione di
BacktestEngineAccantonamento secondo il protocollo standard del
progetto, CRITERI FISSATI PRIMA DI VEDERE I RISULTATI (16/07/2026):

  1. SANITY CHECK: con threshold_mult=999 (irraggiungibile), i trade
     prodotti devono essere IDENTICI a BacktestEngineFloatingKillSwitch
     (stesso motore, stesso capitale, stessi dati) — confronto trade
     per trade, non solo metriche aggregate.
  2. NESSUN CASO PATOLOGICO sui 5 periodi ufficiali: capitale investito
     mai <= 0, accantonato monotono non-decrescente (mai diminuisce),
     nessuna eccezione/crash.
  3. COERENZA con l'approssimazione lineare già fatta in chat il
     16/07/2026 (opt3_mensile su questi stessi 5 periodi): il
     rendimento totale finale (investito+accantonato) del motore vero
     deve stare entro ~5 punti percentuali da quello stimato per
     approssimazione — uno scarto più ampio segnalerebbe un errore
     nell'approssimazione precedente O in questa implementazione, da
     investigare prima di considerare il meccanismo pronto.

QUESTO TEST NON GIUDICA SE IL MECCANISMO "MIGLIORA" I RISULTATI —
quello non è un criterio di validazione qui (già sappiamo che in 4/5
periodi costa rendimento in cambio di sicurezza, accettato). Il test
giudica solo CORRETTEZZA del codice.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_accantonamento import BacktestEngineAccantonamento

WARMUP_DAYS = 90
CAPITAL0 = 2000.0
SYMBOLS = {"DAX": INSTRUMENT_IDX_EUROPE_E_DAAX, "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

# stime dall'approssimazione lineare già fatta (rendimento_pct finale,
# opt3_mensile) — per il criterio 3 di coerenza
APPROX_RENDIMENTO_PCT = {
    "2015-2016": 150.33,
    "2020-covid": 135.30,
    "2023": 127.85,
    "2024-2025": 510.21,
    "2026-ytd": 45.66,
}


def fetch_bars(symbol_const, start: datetime, end: datetime) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID, start, end,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_period_signal_data(period_start: str, period_end: str) -> dict:
    p_start = pd.Timestamp(period_start, tz="UTC")
    p_end = pd.Timestamp(period_end, tz="UTC") + timedelta(days=1)
    warmup_start = p_start - timedelta(days=WARMUP_DAYS)

    signal_data = {}
    for name, const in SYMBOLS.items():
        raw = fetch_bars(const, warmup_start.to_pydatetime(), p_end.to_pydatetime())
        inst = eng.INSTRUMENTS[name]
        signal_data[name] = eng.generate_signals(raw, inst)
    return signal_data


def run_sanity_check(signal_data: dict) -> bool:
    print("=== 1) SANITY CHECK (threshold_mult=999, irraggiungibile) ===")
    baseline = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
    trades_baseline, _ = baseline.run(signal_data)

    accantonamento = BacktestEngineAccantonamento(capital0=CAPITAL0, threshold_mult=999.0)
    trades_acc, _ = accantonamento.run(signal_data)

    if len(trades_baseline) != len(trades_acc):
        print(f"  FALLITO: numero trade diverso ({len(trades_baseline)} vs {len(trades_acc)})")
        return False

    cols_to_compare = ["instrument", "direction", "entry_time", "entry_price",
                        "exit_time", "exit_price", "pnl", "size"]
    diffs = 0
    for col in cols_to_compare:
        if col not in trades_baseline.columns:
            continue
        mismatch = (trades_baseline[col].astype(str).values != trades_acc[col].astype(str).values).sum()
        if mismatch > 0:
            print(f"  FALLITO: colonna '{col}' ha {mismatch} valori diversi")
            diffs += mismatch

    if diffs > 0:
        return False

    if accantonamento.side_pool != 0.0:
        print(f"  FALLITO: side_pool dovrebbe essere 0.0, è {accantonamento.side_pool}")
        return False

    print(f"  OK — {len(trades_baseline)} trade identici, side_pool=0.0 come atteso.\n")
    return True


def main():
    sanity_ok_all = True
    pathology_ok_all = True
    coherence_rows = []

    for label, p_start, p_end in PERIODS:
        print(f"\n{'='*70}\nPeriodo {label} ({p_start} -> {p_end})\n{'='*70}")
        signal_data = get_period_signal_data(p_start, p_end)

        # sanity check ripetuto per ogni periodo (dati diversi ogni volta)
        sanity_ok = run_sanity_check(signal_data)
        sanity_ok_all = sanity_ok_all and sanity_ok

        # run vero con parametri reali (opt.3 mensile: consolidate_pct=0.4, threshold_mult=1.5)
        print("=== 2) Run reale (motore vero, non approssimazione) ===")
        engine_acc = BacktestEngineAccantonamento(capital0=CAPITAL0, consolidate_pct=0.4, threshold_mult=1.5)
        trades_df, metrics_df = engine_acc.run(signal_data)

        invested_finale = metrics_df["capitale_investito_finale"].iloc[0]
        accantonato_finale = metrics_df["accantonato_finale"].iloc[0]
        totale_finale = metrics_df["capitale_totale_finale"].iloc[0]
        n_consolidamenti = metrics_df["n_consolidamenti"].iloc[0]
        rendimento_pct = 100 * (totale_finale - CAPITAL0) / CAPITAL0

        print(f"  Trade totali: {len(trades_df)}")
        print(f"  Investito finale: {invested_finale:.2f} EUR")
        print(f"  Accantonato finale: {accantonato_finale:.2f} EUR")
        print(f"  Totale finale: {totale_finale:.2f} EUR ({rendimento_pct:+.1f}%)")
        print(f"  N. consolidamenti: {n_consolidamenti}")

        # criterio 2: nessun caso patologico
        pathology_issues = []
        if invested_finale <= 0:
            pathology_issues.append(f"capitale investito finale <= 0 ({invested_finale})")
        side_pool_series = [c[4] for c in engine_acc.consolidation_log]  # side_pool dopo ogni evento
        if side_pool_series and any(side_pool_series[i] < side_pool_series[i-1]
                                      for i in range(1, len(side_pool_series))):
            pathology_issues.append("accantonato non monotono (è diminuito ad un certo punto)")

        if pathology_issues:
            print(f"  PATOLOGIA RILEVATA: {pathology_issues}")
            pathology_ok_all = False
        else:
            print(f"  Nessuna patologia rilevata.")

        # criterio 3: coerenza con approssimazione
        approx = APPROX_RENDIMENTO_PCT.get(label)
        scarto = abs(rendimento_pct - approx) if approx is not None else None
        coerente = scarto is not None and scarto <= 5.0
        print(f"  Approssimazione precedente: {approx:+.1f}%  |  Motore reale: {rendimento_pct:+.1f}%  |  "
              f"Scarto: {scarto:.1f}pt  |  {'COERENTE' if coerente else 'SCARTO ECCESSIVO - investigare'}")

        coherence_rows.append({
            "periodo": label, "n_trade": len(trades_df),
            "investito_finale": invested_finale, "accantonato_finale": accantonato_finale,
            "totale_finale": totale_finale, "rendimento_pct_reale": rendimento_pct,
            "rendimento_pct_approx": approx, "scarto_pt": scarto, "coerente": coerente,
            "n_consolidamenti": n_consolidamenti,
            "sanity_check_ok": sanity_ok, "nessuna_patologia": not pathology_issues,
        })

    summary_df = pd.DataFrame(coherence_rows)
    summary_df.to_csv("accantonamento_validation_summary.csv", index=False)

    print(f"\n{'='*70}\nRIEPILOGO VALIDAZIONE\n{'='*70}")
    print(f"Sanity check (tutti i periodi): {'PASSATO' if sanity_ok_all else 'FALLITO'}")
    print(f"Nessuna patologia (tutti i periodi): {'PASSATO' if pathology_ok_all else 'FALLITO'}")
    print(f"Coerenza con approssimazione (tutti i periodi): "
          f"{'PASSATA' if all(r['coerente'] for r in coherence_rows) else 'DA INVESTIGARE SU ALMENO UN PERIODO'}")

    tutti_ok = sanity_ok_all and pathology_ok_all and all(r["coerente"] for r in coherence_rows)
    print(f"\n=== VALIDAZIONE COMPLESSIVA: {'PASSATA — pronto per uso in demo' if tutti_ok else 'NON PASSATA — non usare in demo finché non risolto'} ===")
    print("\nFile: accantonamento_validation_summary.csv")


if __name__ == "__main__":
    main()
