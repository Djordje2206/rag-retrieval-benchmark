"""Glavni meni projekta. Pokretanje: python main.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from demo import pokreni_demo  # noqa: E402
from grafici import napravi_sve  # noqa: E402
from rq1_modeli_chunking import pokreni_rq1  # noqa: E402
from rq2_metrike_udaljenosti import pokreni_rq2  # noqa: E402
from rq3_hnsw_parametri import pokreni_rq3  # noqa: E402

OPCIJE = {
    "1": ("RQ1: model × chunking (72 merenja)", pokreni_rq1),
    "2": ("RQ2: metrike udaljenosti", pokreni_rq2),
    "3": ("RQ3: HNSW parametri", pokreni_rq3),
    "4": ("Napravi grafike i tabele", napravi_sve),
    "5": ("Demo pretraga", pokreni_demo),
}


def main():
    while True:
        _ispisi_meni()
        izbor = input("Izbor > ").strip()
        if izbor == "0":
            break
        if izbor in OPCIJE:
            _, akcija = OPCIJE[izbor]
            akcija()
        else:
            print("Nepoznat izbor.")


def _ispisi_meni():
    print("\nRAG benchmark nad pravilnicima ATVSS Niš")
    for taster, (opis, _) in OPCIJE.items():
        print(f"  {taster}. {opis}")
    print("  0. Izlaz")


if __name__ == "__main__":
    main()
