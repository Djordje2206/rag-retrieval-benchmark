"""
Provera test skupa: da li se ključna fraza svakog pitanja doslovno nalazi u
dokumentu iz koga treba da dođe odgovor. Fraza koja ne postoji nikad ne bi
mogla biti pogođena, pa bi kvarila rezultate.

Pokretanje iz glavnog foldera: python src/proveri_fraze.py
"""

from dokumenti import ucitaj_sve_dokumente
from metrike import je_pogodak
from pitanja import pitanja_sa_odgovorom


def nepostojece_fraze() -> list[dict]:
    dokumenti = ucitaj_sve_dokumente()
    return [
        p
        for p in pitanja_sa_odgovorom()
        if not je_pogodak(dokumenti[p["izvor"]], p["kljucna_fraza"])
    ]


if __name__ == "__main__":
    problematicna = nepostojece_fraze()
    if not problematicna:
        print(f"Sve ključne fraze ({len(pitanja_sa_odgovorom())}) postoje u dokumentima.")
    for p in problematicna:
        print(f"Pitanje {p['id']}: fraza ne postoji u '{p['izvor']}': {p['kljucna_fraza']}")
