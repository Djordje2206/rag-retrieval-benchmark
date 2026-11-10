"""
RQ1: kvalitet pretrage za svaki model i svaku kombinaciju chunkinga
(4 modela × 2 načina sečenja × 3 veličine × 3 preklapanja = 72 merenja).

Rezultat se upisuje posle svakog merenja; ponovno pokretanje preskače već završena.
Pokretanje iz glavnog foldera: python src/rq1_modeli_chunking.py
"""

import time

from baza import napravi_korpus
from chunking import sve_konfiguracije
from embedding import EmbeddingModel
from metrike import izmeri
from modeli import MODELI
from rezultati import sacuvaj_csv, ucitaj_csv

IZLAZNI_CSV = "rq1_modeli_chunking.csv"


def pokreni_rq1():
    redovi = ucitaj_csv(IZLAZNI_CSV)
    zavrsena = {_kljuc(r["model"], r["secenje"], r["velicina"], r["overlap"]) for r in redovi}
    ukupno = len(MODELI) * len(sve_konfiguracije())
    print(f"Već završeno: {len(zavrsena)}/{ukupno} merenja.")

    for opis in MODELI:
        preostale = [
            k
            for k in sve_konfiguracije()
            if _kljuc(opis.ime, k.secenje, k.velicina, k.overlap) not in zavrsena
        ]
        if not preostale:
            continue

        print(f"\nModel: {opis.ime}")
        model = EmbeddingModel(opis)
        for konfiguracija in preostale:
            red = _izmeri_kombinaciju(model, konfiguracija)
            redovi.append(red)
            sacuvaj_csv(IZLAZNI_CSV, redovi)
            print(
                f"  [{len(redovi)}/{ukupno}] {str(konfiguracija):<28} MRR {red['MRR']:.3f}",
                flush=True,
            )


def _izmeri_kombinaciju(model, konfiguracija) -> dict:
    pocetak = time.perf_counter()
    kolekcija, broj_chunkova = napravi_korpus(model, konfiguracija)
    vreme_indeksiranja = time.perf_counter() - pocetak
    m = izmeri(kolekcija, model)
    opis = model.opis

    return {
        "model": opis.ime,
        "opis": opis.opis,
        "dimenzija": opis.dimenzija,
        "secenje": konfiguracija.secenje,
        "velicina": konfiguracija.velicina,
        "overlap": konfiguracija.overlap,
        "broj_chunkova": broj_chunkova,
        "throughput_chunk_s": round(broj_chunkova / vreme_indeksiranja, 1),
        "memorija_mb": round(broj_chunkova * opis.dimenzija * 4 / 1024**2, 2),
        "hit_rate@1": round(m["hit_rate@1"], 4),
        "hit_rate@3": round(m["hit_rate@3"], 4),
        "hit_rate@5": round(m["hit_rate@5"], 4),
        "MRR": round(m["MRR"], 4),
        "latencija_ms": round(m["latencija_ms"], 2),
        "stopa_prepoznavanja_van_teme": round(m["stopa_prepoznavanja_van_teme"], 4),
        "distanca_van_teme": round(m["prosek_distance_van_teme"], 4),
    }


def _kljuc(model, secenje, velicina, overlap):
    return model, secenje, velicina, overlap


if __name__ == "__main__":
    pokreni_rq1()
