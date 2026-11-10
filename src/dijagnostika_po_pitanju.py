"""
Detaljan pregled po pitanju za najbolju konfiguraciju: na kom mestu (od prvih 10)
je pronađen tačan odgovor i koja pitanja sistem promašuje.

Pokretanje iz glavnog foldera: python src/dijagnostika_po_pitanju.py
"""

from collections import Counter

from baza import napravi_korpus, pretrazi
from chunking import NAJBOLJA_KONFIGURACIJA
from embedding import EmbeddingModel
from metrike import rang_prvog_pogotka
from modeli import NAJBOLJI_MODEL
from pitanja import pitanja_sa_odgovorom

K = 10


def pokreni_dijagnostiku():
    model = EmbeddingModel(NAJBOLJI_MODEL)
    kolekcija, _ = napravi_korpus(model, NAJBOLJA_KONFIGURACIJA)

    rangovi = Counter()
    promasena = []
    for pitanje in pitanja_sa_odgovorom():
        rezultati = pretrazi(kolekcija, model, pitanje["pitanje"], K)
        rang = rang_prvog_pogotka(rezultati, pitanje["kljucna_fraza"])
        _ispisi_pitanje(pitanje, rang, rezultati[0])
        if rang is None:
            promasena.append(pitanje)
        else:
            rangovi[rang] += 1

    _ispisi_rezime(rangovi, promasena)


def _ispisi_pitanje(pitanje, rang, najblizi):
    ishod = f"pronađeno na mestu {rang}" if rang else f"nije u prvih {K}"
    print(f"[{pitanje['id']}] ({pitanje['tip']}) {ishod}")
    print(f"    Pitanje:   {pitanje['pitanje']}")
    print(f"    Očekivano: [{pitanje['izvor']}] {pitanje['kljucna_fraza'][:60]}")
    print(f"    Najbliži:  [{najblizi.izvor}] distanca {najblizi.distanca:.3f}\n")


def _ispisi_rezime(rangovi, promasena):
    ukupno = sum(rangovi.values()) + len(promasena)
    kumulativno = 0
    for rang in sorted(rangovi):
        kumulativno += rangovi[rang]
        print(f"Mesto {rang}: {rangovi[rang]} (u prvih {rang}: {kumulativno}/{ukupno})")
    print(f"Nije u prvih {K}: {len(promasena)}/{ukupno}")
    for pitanje in promasena:
        print(f"  - [{pitanje['id']}] {pitanje['pitanje']}")


if __name__ == "__main__":
    pokreni_dijagnostiku()
