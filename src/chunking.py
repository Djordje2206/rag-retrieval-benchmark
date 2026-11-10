"""
Sečenje teksta na chunkove.

Veličina chunka se meri tiktoken (cl100k_base) tokenima, ne tokenima embedding
modela. Za srpsku ćirilicu je 512 cl100k tokena oko 210 tokena modela.
"""

import re
from dataclasses import dataclass

import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

SECENJA = ("standardno", "sentence-aware")
VELICINE = (128, 512, 1024)
OVERLAPI = (0, 10, 20)


@dataclass(frozen=True)
class KonfiguracijaChunkinga:
    secenje: str
    velicina: int
    overlap: int

    def __str__(self) -> str:
        return f"{self.secenje}, {self.velicina} / {self.overlap}%"


NAJBOLJA_KONFIGURACIJA = KonfiguracijaChunkinga("sentence-aware", 512, 10)


def sve_konfiguracije() -> list[KonfiguracijaChunkinga]:
    return [
        KonfiguracijaChunkinga(secenje, velicina, overlap)
        for secenje in SECENJA
        for velicina in VELICINE
        for overlap in OVERLAPI
    ]


def iseci(tekst: str, konfiguracija: KonfiguracijaChunkinga) -> list[str]:
    if konfiguracija.secenje == "standardno":
        return seci_standardno(tekst, konfiguracija.velicina, konfiguracija.overlap)
    if konfiguracija.secenje == "sentence-aware":
        return seci_po_recenicama(tekst, konfiguracija.velicina, konfiguracija.overlap)
    raise ValueError(f"Nepoznat način sečenja: {konfiguracija.secenje}")


def broj_tokena(tekst: str) -> int:
    return len(TOKENIZER.encode(tekst))


def seci_standardno(tekst: str, velicina: int, overlap_procenat: int) -> list[str]:
    """Deli tekst na delove od tačno `velicina` tokena, bez obzira na granice rečenica."""
    tokeni = TOKENIZER.encode(tekst)
    tokena_preklapanja = int(velicina * overlap_procenat / 100)
    korak = velicina - tokena_preklapanja if tokena_preklapanja < velicina else velicina

    return [
        TOKENIZER.decode(tokeni[pocetak : pocetak + velicina])
        for pocetak in range(0, len(tokeni), korak)
    ]


def seci_po_recenicama(tekst: str, ciljna_velicina: int, overlap_procenat: int) -> list[str]:
    """
    Grupiše cele rečenice dok chunk ne dostigne ciljnu veličinu, pa ga zatvara
    na kraju rečenice. Rečenica duža od ciljne veličine čini chunk sama.
    Preklapanje se računa u rečenicama: sledeći chunk počinje poslednjim
    rečenicama prethodnog.
    """
    recenice = podeli_na_recenice(tekst)
    tokena_po_recenici = [broj_tokena(r) for r in recenice]

    chunkovi = []
    tekuci = []
    tokena_u_tekucem = 0
    i = 0
    while i < len(recenice):
        prepunjen = tekuci and tokena_u_tekucem + tokena_po_recenici[i] > ciljna_velicina
        if prepunjen:
            chunkovi.append(_spoji(recenice, tekuci))
            tekuci = _recenice_za_preklapanje(
                tekuci, tokena_po_recenici, ciljna_velicina, overlap_procenat
            )
            tokena_u_tekucem = sum(tokena_po_recenici[j] for j in tekuci)
            continue

        tekuci.append(i)
        tokena_u_tekucem += tokena_po_recenici[i]
        i += 1

    if tekuci:
        chunkovi.append(_spoji(recenice, tekuci))
    return chunkovi


def podeli_na_recenice(tekst: str) -> list[str]:
    jedan_red = re.sub(r"\s+", " ", tekst).strip()
    delovi = re.split(r"(?<=[\.\!\?])\s+", jedan_red)
    return [d.strip() for d in delovi if d.strip()]


def _recenice_za_preklapanje(tekuci, tokena_po_recenici, ciljna_velicina, overlap_procenat):
    # Preklapanje mora da ostavi bar jednu novu rečenicu i ne sme samo da popuni
    # ciljnu veličinu, inače bi se isti chunk pravio u beskonačnoj petlji.
    if overlap_procenat == 0 or len(tekuci) < 2:
        return []
    broj = max(1, int(len(tekuci) * overlap_procenat / 100))
    broj = min(broj, len(tekuci) - 1)
    preklapanje = tekuci[-broj:]
    if sum(tokena_po_recenici[j] for j in preklapanje) >= ciljna_velicina:
        return []
    return preklapanje


def _spoji(recenice, indeksi) -> str:
    return " ".join(recenice[j] for j in indeksi)
