"""Čitanje i pisanje rezultata eksperimenata (CSV) i putanje do foldera sa rezultatima."""

import csv
import os

FOLDER_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER_MERENJA = os.path.join(FOLDER_PROJEKTA, "rezultati", "raw")
FOLDER_SLIKA = os.path.join(FOLDER_PROJEKTA, "rezultati", "figures")


def sacuvaj_csv(ime_fajla: str, redovi: list[dict]) -> str:
    os.makedirs(FOLDER_MERENJA, exist_ok=True)
    putanja = os.path.join(FOLDER_MERENJA, ime_fajla)
    with open(putanja, "w", newline="", encoding="utf-8") as f:
        pisac = csv.DictWriter(f, fieldnames=list(redovi[0].keys()))
        pisac.writeheader()
        pisac.writerows(redovi)
    return putanja


def ucitaj_csv(ime_fajla: str) -> list[dict]:
    """Vraća redove sa brojevima kao int/float. Ako fajl ne postoji, vraća praznu listu."""
    putanja = os.path.join(FOLDER_MERENJA, ime_fajla)
    if not os.path.exists(putanja):
        return []
    with open(putanja, encoding="utf-8") as f:
        return [{k: _u_broj(v) for k, v in red.items()} for red in csv.DictReader(f)]


def _u_broj(vrednost: str):
    for tip in (int, float):
        try:
            return tip(vrednost)
        except ValueError:
            pass
    return vrednost
