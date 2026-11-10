"""
Interaktivna pretraga nad najboljom konfiguracijom.

Za svaki pronađeni chunk prikazuje najrelevantniju stavku kao odgovor, a ispod
početak chunka kao kontekst. Izdvajanje stavke služi samo za prikaz; merenja u
eksperimentima uvek rade nad celim chunkom.
"""

import re

import numpy as np

from baza import napravi_korpus, pretrazi
from chunking import NAJBOLJA_KONFIGURACIJA
from embedding import EmbeddingModel
from metrike import PRAG_NEPOZNATO
from modeli import NAJBOLJI_MODEL

BROJ_REZULTATA = 3
PRAG_VISOKE_SIGURNOSTI = 0.50
MAKS_DUZINA_ODGOVORA = 280
DUZINA_KONTEKSTA = 200
MIN_DUZINA_STAVKE = 15
DUGA_RECENICA = 150
KRAJ = ("izlaz", "exit", "quit", "")


def pokreni_demo():
    print(f"Učitavam {NAJBOLJI_MODEL.ime} i pravim bazu ({NAJBOLJA_KONFIGURACIJA})...")
    model = EmbeddingModel(NAJBOLJI_MODEL)
    kolekcija, broj_chunkova = napravi_korpus(model, NAJBOLJA_KONFIGURACIJA)
    print(f"Baza spremna: {broj_chunkova} chunkova.")
    print("Postavi pitanje o pravilnicima ili zakonu ('izlaz' za kraj).\n")

    while (pitanje := input("PITANJE > ").strip()).lower() not in KRAJ:
        _odgovori(kolekcija, model, pitanje)


def _odgovori(kolekcija, model, pitanje):
    upit = model.embeduj_upit(pitanje)
    rezultati = pretrazi(kolekcija, model, pitanje, BROJ_REZULTATA)

    for i, r in enumerate(rezultati, start=1):
        odgovor = _najrelevantnija_stavka(model, upit, r.tekst)
        kontekst = r.tekst.strip().replace("\n", " ")[:DUZINA_KONTEKSTA]
        sigurnost = _sigurnost(r.distanca)
        print(f"\n  {i}. [{r.izvor}] distanca {r.distanca:.3f} (sigurnost: {sigurnost})")
        print(f"     ODGOVOR: {_skrati(odgovor, MAKS_DUZINA_ODGOVORA)}")
        print(f"     (kontekst: {kontekst}...)")

    if rezultati[0].distanca >= PRAG_NEPOZNATO:
        print("\n  Sistem nije siguran; pitanje možda nije pokriveno dokumentima.")
    print()


def _sigurnost(distanca: float) -> str:
    if distanca < PRAG_VISOKE_SIGURNOSTI:
        return "visoka"
    if distanca < PRAG_NEPOZNATO:
        return "srednja"
    return "niska"


def _najrelevantnija_stavka(model, upit, tekst: str) -> str:
    stavke = _podeli_na_stavke(tekst)
    if len(stavke) <= 1:
        return stavke[0] if stavke else tekst.strip()

    embeddingi = model.embeduj_dokumente(stavke)
    slicnosti = (
        embeddingi @ upit / (np.linalg.norm(embeddingi, axis=1) * np.linalg.norm(upit) + 1e-8)
    )
    return stavke[int(np.argmax(slicnosti))]


def _podeli_na_stavke(tekst: str) -> list[str]:
    # Nabrajanja u pravilnicima ("i to: 1) ...; 2) ...") su gramatički jedna
    # rečenica, pa se duge rečenice dodatno dele po tački-zarezu.
    stavke = []
    for deo in re.split(r"(?<=[.!?])\s+", tekst.strip()):
        recenica = deo.strip()
        if len(recenica) > DUGA_RECENICA and ";" in recenica:
            stavke.extend(d.strip(" ;") for d in re.split(r";\s*", recenica))
        else:
            stavke.append(recenica)
    return [s for s in stavke if len(s) > MIN_DUZINA_STAVKE]


def _skrati(tekst: str, maks_duzina: int) -> str:
    if len(tekst) <= maks_duzina:
        return tekst
    return tekst[:maks_duzina].rsplit(" ", 1)[0] + "..."


if __name__ == "__main__":
    pokreni_demo()
