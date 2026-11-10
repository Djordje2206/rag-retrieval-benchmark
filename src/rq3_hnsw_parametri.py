"""
RQ3: uticaj HNSW parametara (M, ef_construction) na kvalitet i brzinu.

Koristi najbolji model i chunking iz RQ1. Odvojeno se mere gradnja indeksa,
sama pretraga u bazi (bez embedovanja pitanja) i recall@5 u odnosu na tačnu
(brute-force) pretragu.

Pokretanje iz glavnog foldera: python src/rq3_hnsw_parametri.py
"""

import time

import numpy as np

from baza import napravi_chunkove, napravi_kolekciju
from chunking import NAJBOLJA_KONFIGURACIJA
from embedding import EmbeddingModel
from metrike import izmeri
from modeli import NAJBOLJI_MODEL
from pitanja import PITANJA
from rezultati import sacuvaj_csv

IZLAZNI_CSV = "rq3_hnsw_parametri.csv"
M_VREDNOSTI = (8, 16, 32)
EF_CONSTRUCTION_VREDNOSTI = (100, 200)
K = 5
PONAVLJANJA_PRETRAGE = 20


def pokreni_rq3():
    print(f"RQ3: {NAJBOLJI_MODEL.ime}, {NAJBOLJA_KONFIGURACIJA}")
    model = EmbeddingModel(NAJBOLJI_MODEL)
    chunkovi = napravi_chunkove(NAJBOLJA_KONFIGURACIJA)
    embeddingi = model.embeduj_dokumente([c.tekst for c in chunkovi])

    upiti, embedovanje_upita_ms = _embeduj_pitanja_jedno_po_jedno(model)

    tacni_top_k = _tacna_pretraga([c.id for c in chunkovi], embeddingi, upiti)

    redovi = []
    for M in M_VREDNOSTI:
        for ef in EF_CONSTRUCTION_VREDNOSTI:
            pocetak = time.perf_counter()
            kolekcija = napravi_kolekciju(chunkovi, embeddingi, M=M, ef_construction=ef)
            vreme_indeksa_ms = (time.perf_counter() - pocetak) * 1000
            m = izmeri(kolekcija, model)

            redovi.append(
                {
                    "M": M,
                    "ef_construction": ef,
                    "broj_chunkova": len(chunkovi),
                    "vreme_gradnje_indeksa_ms": round(vreme_indeksa_ms, 1),
                    "latencija_pretrage_ms": round(_latencija_pretrage_ms(kolekcija, upiti), 3),
                    "latencija_embedovanja_upita_ms": round(embedovanje_upita_ms, 1),
                    "recall@5_vs_tacna": round(_recall(kolekcija, upiti, tacni_top_k), 4),
                    "hit_rate@1": round(m["hit_rate@1"], 4),
                    "hit_rate@3": round(m["hit_rate@3"], 4),
                    "hit_rate@5": round(m["hit_rate@5"], 4),
                    "MRR": round(m["MRR"], 4),
                }
            )
            print(f"  M={M:<2} ef={ef:<3} MRR {m['MRR']:.4f}")

    print(f"Sačuvano: {sacuvaj_csv(IZLAZNI_CSV, redovi)}")


def _embeduj_pitanja_jedno_po_jedno(model):
    """Vraća embeddinge svih pitanja i prosečno vreme embedovanja jednog pitanja (ms)."""
    pocetak = time.perf_counter()
    upiti = np.array([model.embeduj_upit(p["pitanje"]) for p in PITANJA])
    return upiti, (time.perf_counter() - pocetak) * 1000 / len(PITANJA)


def _tacna_pretraga(ids, embeddingi, upiti) -> list[set]:
    dokumenti_n = embeddingi / np.linalg.norm(embeddingi, axis=1, keepdims=True)
    upiti_n = upiti / np.linalg.norm(upiti, axis=1, keepdims=True)
    niz_ids = np.array(ids)
    return [set(niz_ids[np.argsort(-(dokumenti_n @ upit))[:K]]) for upit in upiti_n]


def _latencija_pretrage_ms(kolekcija, upiti) -> float:
    lista_upita = upiti.tolist()
    pocetak = time.perf_counter()
    for _ in range(PONAVLJANJA_PRETRAGE):
        for upit in lista_upita:
            kolekcija.query(query_embeddings=[upit], n_results=K)
    return (time.perf_counter() - pocetak) * 1000 / (PONAVLJANJA_PRETRAGE * len(lista_upita))


def _recall(kolekcija, upiti, tacni_top_k) -> float:
    hnsw_top_k = kolekcija.query(query_embeddings=upiti.tolist(), n_results=K)["ids"]
    return float(
        np.mean([len(set(h) & t) / K for h, t in zip(hnsw_top_k, tacni_top_k, strict=True)])
    )


if __name__ == "__main__":
    pokreni_rq3()
