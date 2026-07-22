"""
ftse_regime_candidates_persistence.py — Filone "identificazione regime
FTSE100", angolo concettualmente diverso da ATR%/VR/autocorrelazione
(esauriti il 22/07/2026, vedi 00_CURRENT_STATE.md sez.14 e
03_CLOSED_RESEARCH_REGISTRY.md). Testa in parallelo 3 nuovi candidati,
stesso protocollo di validazione già usato per ATR%/correlazione DAX-
FTSE100 (persistenza su finestre settimanali NON sovrapposte — il
run-length su barre consecutive è un artefatto, vedi lezione 22/07 —
soglie di bucket fissate SOLO sul train, applicate poi all'holdout
2024-2025 senza ricalibrarle, per un test di generalizzazione vero).

Candidati:
  1. corr_ftse_gbpusd (7g/21g): correlazione rolling tra rendimenti
     FTSE100 e GBPUSD. Ipotesi: FTSE100 ha forte quota di ricavi esteri
     (multinazionali/minerarie/energetiche), storicamente sensibile a
     GBP — magnitude/co-movimento, stesso tipo di ATR%/corr DAX-FTSE
     che hanno persistito.
  2. ratio_dax_ftse_mom (7g/21g): rendimento log del rapporto
     close_DAX/close_FTSE100 su finestra rolling — "chi guida chi",
     non "si muovono insieme". Variabile DIREZIONALE (come VR/
     autocorrelazione, che NON sono persistite standalone) — testata
     comunque perché è un angolo mai provato, non una riproposizione
     delle stesse variabili con soglie diverse.
  3. corr_ftse_gold (7g/21g): correlazione rolling tra rendimenti
     FTSE100 e GOLD. FTSE100 sovrappesato in minerarie/energia rispetto
     a DAX — magnitude/co-movimento.

Bucket: terzili (bassa/media/alta) per i candidati magnitude-type (1,3),
binario (su/giù) per il candidato direzionale (2) — stessa logica delle
3 categorie di stato già usate per ATR%/correlazione in sez.14.

Persistenza per stato: P(stato settimana t+1 == stato settimana t |
stato settimana t == s), confrontata con base rate (1/3 per terzili,
1/2 per binario). Train = tutti i periodi tranne 2024-2025 (soglie
bucket fissate SOLO qui). Holdout = 2024-2025 (soglie applicate senza
ricalibrare).

IMPORTANTE (regola del progetto): stampa SOLO risultati aggregati
(percentuali di persistenza per stato/periodo), MAI serie temporali o
righe individuali — coerente con "mai testo letterale >~50 righe di
dati grezzi in chat".

NOTA: script non ancora eseguito una volta prima della consegna (limite
ambiente di generazione) — verificarne l'output al primo run reale in
Actions prima di trattare i risultati come definitivi, come da regola
obbligatoria del progetto ("mai consegnare uno script mai eseguito").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ohlc_data_source import get_ohlc

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-14"),
}
HOLDOUT_LABEL = "2024-2025"


def in_periods(ts_index: pd.DatetimeIndex, labels: list[str]) -> pd.Series:
    mask = pd.Series(False, index=ts_index)
    for lbl in labels:
        start, end = PERIODS[lbl]
        mask |= (ts_index >= pd.Timestamp(start, tz="UTC")) & (ts_index < pd.Timestamp(end, tz="UTC"))
    return mask


def rolling_corr(a: pd.Series, b: pd.Series, window: str) -> pd.Series:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    return aligned["a"].rolling(window).corr(aligned["b"]).reindex(a.index)


def weekly_nonoverlap(series: pd.Series) -> pd.Series:
    """Ultimo valore per settimana calendariale (non sovrapposta) — stesso
    principio del test di persistenza usato per ATR%/correlazione in sez.14:
    il run-length su barre rolling consecutive è un artefatto, qui invece
    ogni punto rappresenta una settimana distinta."""
    return series.resample("W").last().dropna()


def tercile_thresholds(train_values: pd.Series) -> tuple[float, float]:
    q1, q2 = train_values.quantile([1 / 3, 2 / 3])
    return q1, q2


def bucket_tercile(values: pd.Series, q1: float, q2: float) -> pd.Series:
    return pd.cut(values, bins=[-np.inf, q1, q2, np.inf], labels=["bassa", "media", "alta"])


def bucket_binary(values: pd.Series) -> pd.Series:
    return pd.Series(np.where(values >= 0, "su", "giu"), index=values.index)


def persistence_by_state(states: pd.Series) -> dict:
    """Per ogni settimana t con stato s, guarda lo stato alla settimana
    t+1 (prossima riga della serie settimanale, NON necessariamente 7
    giorni esatti se manca un dato — accettabile, sono comunque punti
    settimanali distinti e non sovrapposti)."""
    s = states.dropna()
    if len(s) < 3:
        return {}
    cur = s.iloc[:-1].values
    nxt = s.iloc[1:].values
    out = {}
    for state in s.cat.categories if hasattr(s, "cat") else sorted(set(s)):
        mask = cur == state
        n = mask.sum()
        if n == 0:
            out[state] = (np.nan, 0)
            continue
        persist_rate = (nxt[mask] == state).mean()
        out[state] = (persist_rate, int(n))
    return out


def run_candidate(name: str, magnitude_series: dict, is_directional: bool):
    """magnitude_series: dict window_label -> pd.Series (indice=timestamp,
    valore=candidato). is_directional: True per bucket binario, False per
    terzili."""
    print(f"\n{'=' * 70}\nCANDIDATO: {name}\n{'=' * 70}")

    for window_label, series in magnitude_series.items():
        weekly = weekly_nonoverlap(series)
        idx = weekly.index

        train_mask = in_periods(idx, [p for p in PERIODS if p != HOLDOUT_LABEL])
        holdout_mask = in_periods(idx, [HOLDOUT_LABEL])

        train_vals = weekly[train_mask]
        holdout_vals = weekly[holdout_mask]

        if len(train_vals) < 10 or len(holdout_vals) < 5:
            print(f"  [{window_label}] Dati insufficienti (train={len(train_vals)}, "
                  f"holdout={len(holdout_vals)}) — salto.")
            continue

        if is_directional:
            train_states = bucket_binary(train_vals)
            holdout_states = bucket_binary(holdout_vals)
            base_rate = 0.5
        else:
            q1, q2 = tercile_thresholds(train_vals)
            train_states = bucket_tercile(train_vals, q1, q2)
            holdout_states = bucket_tercile(holdout_vals, q1, q2)
            base_rate = 1 / 3

        train_persist = persistence_by_state(train_states)
        holdout_persist = persistence_by_state(holdout_states)

        print(f"\n  --- finestra {window_label} (base rate: {base_rate*100:.1f}%) ---")
        print(f"  train (n settimane={len(train_vals)}):")
        for state, (rate, n) in sorted(train_persist.items()):
            delta = "" if np.isnan(rate) else f" ({(rate - base_rate) * 100:+.1f}pt vs base rate)"
            print(f"    {state:8s}: persistenza={rate*100:5.1f}% n={n}{delta}" if not np.isnan(rate)
                  else f"    {state:8s}: n=0")
        print(f"  holdout {HOLDOUT_LABEL} (n settimane={len(holdout_vals)}):")
        for state, (rate, n) in sorted(holdout_persist.items()):
            delta = "" if np.isnan(rate) else f" ({(rate - base_rate) * 100:+.1f}pt vs base rate)"
            print(f"    {state:8s}: persistenza={rate*100:5.1f}% n={n}{delta}" if not np.isnan(rate)
                  else f"    {state:8s}: n=0")


def main():
    import os
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]

    print("Scarico OHLC (DAX, FTSE100, GOLD, GBPUSD) via ohlc_data_source...")
    dax = get_ohlc("DAX", account_id, token).set_index("timestamp")["close"]
    ftse = get_ohlc("FTSE100", account_id, token).set_index("timestamp")["close"]
    gold = get_ohlc("GOLD", account_id, token).set_index("timestamp")["close"]
    gbpusd = get_ohlc("GBPUSD", account_id, token).set_index("timestamp")["close"]

    ret_ftse = ftse.pct_change()
    ret_gbpusd = gbpusd.pct_change()
    ret_gold = gold.pct_change()

    # --- Candidato 1: corr FTSE-GBPUSD ---
    corr_ftse_gbpusd = {
        "7d": rolling_corr(ret_ftse, ret_gbpusd, "7D"),
        "21d": rolling_corr(ret_ftse, ret_gbpusd, "21D"),
    }
    run_candidate("corr_ftse_gbpusd (magnitude, terzili)", corr_ftse_gbpusd, is_directional=False)

    # --- Candidato 2: momentum ratio DAX/FTSE ---
    aligned = pd.concat([dax.rename("dax"), ftse.rename("ftse")], axis=1, sort=True).dropna()
    log_ratio = np.log(aligned["dax"] / aligned["ftse"])
    ratio_mom = {
        "7d": log_ratio.diff(1).rolling("7D").sum(),  # approssimazione: somma variazioni su finestra
        "21d": log_ratio.diff(1).rolling("21D").sum(),
    }
    run_candidate("ratio_dax_ftse_momentum (direzionale, binario)", ratio_mom, is_directional=True)

    # --- Candidato 3: corr FTSE-GOLD ---
    corr_ftse_gold = {
        "7d": rolling_corr(ret_ftse, ret_gold, "7D"),
        "21d": rolling_corr(ret_ftse, ret_gold, "21D"),
    }
    run_candidate("corr_ftse_gold (magnitude, terzili)", corr_ftse_gold, is_directional=False)

    print("\n" + "=" * 70)
    print("Completato. Nessuna scrittura su D1 — solo stampa aggregata.")
    print("=" * 70)


if __name__ == "__main__":
    main()
