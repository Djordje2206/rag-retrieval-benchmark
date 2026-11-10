"""Učitavanje dokumenata od kojih se pravi baza znanja."""

import os

import pymupdf

FOLDER_PODATAKA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# izvor -> ime fajla u folderu data/
DOKUMENTI = {
    "zakon": "zakon.pdf",
    "praksa": "praksa.txt",
    "vannastavne": "vannastavne.txt",
}


def ucitaj_dokument(izvor: str) -> str:
    putanja = os.path.join(FOLDER_PODATAKA, DOKUMENTI[izvor])
    if putanja.endswith(".pdf"):
        return _procitaj_pdf(putanja)
    return _procitaj_txt(putanja)


def ucitaj_sve_dokumente() -> dict[str, str]:
    return {izvor: ucitaj_dokument(izvor) for izvor in DOKUMENTI}


def _procitaj_pdf(putanja: str) -> str:
    with pymupdf.open(putanja) as dokument:
        return "".join(strana.get_text() for strana in dokument)


def _procitaj_txt(putanja: str) -> str:
    with open(putanja, encoding="utf-8") as f:
        return f.read()
