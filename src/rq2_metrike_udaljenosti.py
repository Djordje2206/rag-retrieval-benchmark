"""
RQ2: uticaj metrike udaljenosti (kosinusna, euklidska, skalarni proizvod) na kvalitet.

Koristi najbolji model i chunking iz RQ1. Embeddingi se računaju jednom, pa se
metrike razlikuju samo po indeksu. Beleži se i prosečna dužina (norma) vektora:
za vektore dužine 1 sve tri metrike daju isti redosled rezultata.

Pokretanje iz glavnog foldera: python src/rq2_metrike_udaljenosti.py
"""

import time

import numpy as np

from baza import napravi_chunkove, napravi_kolekciju
from chunking import NAJBOLJA_KONFIGURACIJA
from embedding import EmbeddingModel
from metrike import izmeri
from modeli import NAJBOLJI_MODEL
from rezultati import sacuvaj_csv

IZLAZNI_CSV = "rq2_metrike_udaljenosti.csv"

METRIKE = {
    "cosine": "Kosinusna",
    "l2": "Euklidska L2",
    "ip": "Skalarni proizvod",
}


def pokreni_rq2():
    print(f"RQ2: {NAJBOLJI_MODEL.ime}, {NAJBOLJA_KONFIGURACIJA}")
    model = EmbeddingModel(NAJBOLJI_MODEL)
    chunkovi = napravi_chunkove(NAJBOLJA_KONFIGURACIJA)

    pocetak = time.perf_counter()
    embeddingi = model.embeduj_dokumente([c.tekst for c in chunkovi])
    vreme_embedovanja = time.perf_counter() - pocetak
    prosecna_norma = float(np.linalg.norm(embeddingi, axis=1).mean())

    redovi = []
    for kod, naziv in METRIKE.items():
        pocetak = time.perf_counter()
        kolekcija = napravi_kolekciju(chunkovi, embeddingi, metrika=kod)
        vreme_indeksa_ms = (time.perf_counter() - pocetak) * 1000
        m = izmeri(kolekcija, model)

        redovi.append(
            {
                "metrika_kod": kod,
                "metrika_naziv": naziv,
                "broj_chunkova": len(chunkovi),
                "prosecna_norma_vektora": round(prosecna_norma, 4),
                "vreme_embedovanja_s": round(vreme_embedovanja, 1),
                "vreme_gradnje_indeksa_ms": round(vreme_indeksa_ms, 1),
                "hit_rate@1": round(m["hit_rate@1"], 4),
                "hit_rate@3": round(m["hit_rate@3"], 4),
                "hit_rate@5": round(m["hit_rate@5"], 4),
                "MRR": round(m["MRR"], 4),
                "latencija_ms": round(m["latencija_ms"], 2),
            }
        )
        print(f"  {naziv:<18} MRR {m['MRR']:.4f}")

    print(f"Sačuvano: {sacuvaj_csv(IZLAZNI_CSV, redovi)}")


if __name__ == "__main__":
    pokreni_rq2()
