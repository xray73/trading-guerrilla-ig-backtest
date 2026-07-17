"""
orb_adx_signals.py — Ramo esplorativo: segnale Opening Range Breakout
con filtro ADX(14) al posto del VWAP (bloccato da mancanza di volume
affidabile su DAX/FTSE100 cash, verificato 17/07/2026).

SPECIFICA CONFERMATA IN CHAT IL 17/07/2026:
  1. Range di apertura = prima barra da 30min della sessione, ancorata
     all'apertura della BORSA CASH sottostante (non l'orario esteso
     CFD IG): Xetra 09:00 Europe/Berlin (DAX), LSE 08:00 Europe/London
     (FTSE100). Gestione ora legale automatica via zoneinfo, non
     offset fisso — l'apertura reale si sposta di un'ora tra
     inverno/estate e la sessione deve seguirla.
  2. Filtro: ADX(14) CONTINUO (stessa formula Wilder di engine.py, non
     un ADX isolato sulla singola barra — non calcolabile) letto alla
     CHIUSURA della barra di apertura. Soglia 20 (stessa di Variante 6,
     punto di partenza — parametro da testare, non ancora ottimizzato).
     Se ADX(14) a quel punto è sotto soglia, il giorno non genera
     segnali ORB.
  3. Direzione decisa dal breakout stesso (no filtro EMA direzionale
     come in Variante 6): rottura sopra il massimo della barra di
     apertura -> long, sotto il minimo -> short.
  4. REGOLA WHIPSAW: un solo trade per strumento al giorno. Il segnale
     scatta SOLO sulla prima barra post-apertura che rompe il range
     nella prima direzione valida; ignorati tutti i breakout successivi
     nello stesso giorno, in entrambe le direzioni.

Riusa atr_wilder/adx_wilder/wilder_smooth da engine.py (import diretto,
NESSUNA duplicazione di formula, NESSUNA modifica a engine.py).

Output: stesso schema di eng.generate_signals() — colonna 'signal'
('long'/'short'/None) più le colonne indicatore già presenti (atr, adx)
necessarie al motore per il sizing — così BacktestEngineFloatingKillSwitch
(o qualunque sua sottoclasse, es. accantonamento) può consumare questo
segnale SENZA NESSUNA MODIFICA al motore stesso.

STATO: prima implementazione, non ancora testata. Vedi
orb_adx_feasibility_test.py per il primo controllo (fattibilità:
quanti trade/anno genera).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

import engine as eng

SESSION_OPEN = {
    "DAX": {"tz": ZoneInfo("Europe/Berlin"), "hour": 9, "minute": 0},
    "FTSE100": {"tz": ZoneInfo("Europe/London"), "hour": 8, "minute": 0},
}

ADX_THRESHOLD_DEFAULT = 20.0


def _session_open_utc_for_date(date, instrument: str) -> pd.Timestamp:
    cfg = SESSION_OPEN[instrument]
    local_dt = pd.Timestamp(date.year, date.month, date.day, cfg["hour"], cfg["minute"], tz=cfg["tz"])
    return local_dt.tz_convert("UTC")


def generate_orb_adx_signals(df: pd.DataFrame, inst: eng.InstrumentConfig,
                              instrument_name: str, adx_threshold: float = ADX_THRESHOLD_DEFAULT,
                              p: eng.ChartaParams = eng.PARAMS) -> pd.DataFrame:
    """df: colonne timestamp, open, high, low, close (stesso formato di
    eng.generate_signals). instrument_name: 'DAX' o 'FTSE100' (per
    l'orario di sessione). Ritorna df con colonne atr, adx, signal —
    stesso schema consumabile dal motore esistente."""
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    # riuso diretto delle formule Wilder già validate in engine.py
    out["atr"] = eng.atr_wilder(out, p.atr_period)
    out["adx"] = eng.adx_wilder(out, p.adx_period)

    out["date"] = out["timestamp"].dt.date
    out["signal"] = None

    for date, day_rows in out.groupby("date"):
        date_ts = pd.Timestamp(date)
        if date_ts.weekday() >= 5:  # 5=sabato, 6=domenica — Xetra/LSE non aprono nel weekend
            continue

        session_open = _session_open_utc_for_date(date_ts, instrument_name)
        day_bars = day_rows[day_rows["timestamp"] >= session_open]
        if day_bars.empty:
            continue

        opening_idx = day_bars.index[0]
        opening_bar = out.loc[opening_idx]
        range_high = opening_bar["high"]
        range_low = opening_bar["low"]
        adx_at_open = opening_bar["adx"]

        if pd.isna(adx_at_open) or adx_at_open <= adx_threshold:
            continue  # giorno non armato, ADX insufficiente alla chiusura della barra di apertura

        # cerca il PRIMO breakout valido tra le barre successive alla barra di apertura, stesso giorno
        remaining = day_bars.loc[day_bars.index > opening_idx]
        for idx, bar in remaining.iterrows():
            if bar["close"] > range_high:
                out.at[idx, "signal"] = "long"
                break
            elif bar["close"] < range_low:
                out.at[idx, "signal"] = "short"
                break
            # altrimenti nessun breakout ancora su questa barra, continua a scorrere il giorno

    out = out.drop(columns=["date"])
    return out


if __name__ == "__main__":
    import sys
    print("Questo file va importato come modulo. Vedi orb_adx_feasibility_test.py per il primo test.")
    sys.exit(0)
