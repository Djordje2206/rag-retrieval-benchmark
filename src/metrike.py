"""
Merenje kvaliteta pretrage nad test pitanjima.

Pogodak: vraćeni chunk sadrži ključnu frazu tačnog odgovora (ne samo pravi dokument).
Latencija obuhvata embedovanje pitanja i pretragu u bazi.
"""

import re
import time

from baza import pretrazi
from pitanja import pitanja_bez_odgovora, pitanja_sa_odgovorom

K_VREDNOSTI = (1, 3, 5)

# Udaljenost najbližeg chunka iznad koje sistem smatra da odgovor ne postoji.
# Izabrana na osnovu merenja sa bge-m3: tačna pitanja ~0.45-0.55, van teme ~0.6+.
PRAG_NEPOZNATO = 0.60


def izmeri(kolekcija, model) -> dict:
    rangovi, latencije_sa = _rangovi_tacnih_odgovora(kolekcija, model)
    distance, latencije_bez = _distance_pitanja_bez_odgovora(kolekcija, model)
    latencije = latencije_sa + latencije_bez

    rezultat = {f"hit_rate@{k}": _udeo_u_top_k(rangovi, k) for k in K_VREDNOSTI}
    rezultat["MRR"] = sum(1 / r for r in rangovi if r is not None) / len(rangovi)
    rezultat["latencija_ms"] = sum(latencije) / len(latencije)
    odbijena = sum(d >= PRAG_NEPOZNATO for d in distance)
    rezultat["stopa_prepoznavanja_van_teme"] = odbijena / len(distance)
    rezultat["prosek_distance_van_teme"] = sum(distance) / len(distance)
    return rezultat


def je_pogodak(tekst_chunka: str, kljucna_fraza: str) -> bool:
    # Razmaci se normalizuju jer PDF ubacuje prelome redova usred fraza.
    return _normalizuj(kljucna_fraza) in _normalizuj(tekst_chunka)


def rang_prvog_pogotka(rezultati, kljucna_fraza: str):
    for rang, rezultat in enumerate(rezultati, start=1):
        if je_pogodak(rezultat.tekst, kljucna_fraza):
            return rang
    return None


def _rangovi_tacnih_odgovora(kolekcija, model):
    rangovi, latencije = [], []
    for pitanje in pitanja_sa_odgovorom():
        rezultati, ms = _pretrazi_sa_merenjem(
            kolekcija, model, pitanje["pitanje"], max(K_VREDNOSTI)
        )
        rangovi.append(rang_prvog_pogotka(rezultati, pitanje["kljucna_fraza"]))
        latencije.append(ms)
    return rangovi, latencije


def _distance_pitanja_bez_odgovora(kolekcija, model):
    distance, latencije = [], []
    for pitanje in pitanja_bez_odgovora():
        rezultati, ms = _pretrazi_sa_merenjem(kolekcija, model, pitanje["pitanje"], 1)
        distance.append(rezultati[0].distanca)
        latencije.append(ms)
    return distance, latencije


def _pretrazi_sa_merenjem(kolekcija, model, pitanje, k):
    pocetak = time.perf_counter()
    rezultati = pretrazi(kolekcija, model, pitanje, k)
    return rezultati, (time.perf_counter() - pocetak) * 1000


def _udeo_u_top_k(rangovi, k) -> float:
    return sum(1 for r in rangovi if r is not None and r <= k) / len(rangovi)


def _normalizuj(tekst: str) -> str:
    return re.sub(r"\s+", " ", tekst).strip()
