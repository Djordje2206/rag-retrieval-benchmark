"""Pravljenje ChromaDB kolekcije od dokumenata i pretraga."""

import contextlib
from dataclasses import dataclass

import chromadb
from chromadb.errors import NotFoundError

from chunking import KonfiguracijaChunkinga, iseci
from dokumenti import ucitaj_sve_dokumente
from embedding import EmbeddingModel

IME_KOLEKCIJE = "korpus"


@dataclass(frozen=True)
class Chunk:
    id: str
    izvor: str
    tekst: str


@dataclass(frozen=True)
class Rezultat:
    izvor: str
    tekst: str
    distanca: float


def napravi_chunkove(konfiguracija: KonfiguracijaChunkinga) -> list[Chunk]:
    chunkovi = []
    for izvor, tekst in ucitaj_sve_dokumente().items():
        for i, deo in enumerate(iseci(tekst, konfiguracija)):
            chunkovi.append(Chunk(f"{izvor}_{i}", izvor, deo))
    return chunkovi


def napravi_korpus(model: EmbeddingModel, konfiguracija: KonfiguracijaChunkinga):
    """Seče dokumente, embeduje chunkove i pravi kolekciju. Vraća (kolekcija, broj_chunkova)."""
    chunkovi = napravi_chunkove(konfiguracija)
    embeddingi = model.embeduj_dokumente([c.tekst for c in chunkovi])
    return napravi_kolekciju(chunkovi, embeddingi), len(chunkovi)


def napravi_kolekciju(
    chunkovi: list[Chunk], embeddingi, metrika="cosine", M=16, ef_construction=100
):
    """Pravi kolekciju od već izračunatih embeddinga i gradi HNSW indeks."""
    klijent = chromadb.Client()
    with contextlib.suppress(NotFoundError):
        klijent.delete_collection(IME_KOLEKCIJE)

    kolekcija = klijent.create_collection(
        name=IME_KOLEKCIJE,
        metadata={
            "hnsw:space": metrika,
            "hnsw:M": M,
            "hnsw:construction_ef": ef_construction,
        },
    )
    kolekcija.add(
        ids=[c.id for c in chunkovi],
        documents=[c.tekst for c in chunkovi],
        metadatas=[{"izvor": c.izvor} for c in chunkovi],
        embeddings=embeddingi.tolist(),
    )
    return kolekcija


def pretrazi(kolekcija, model: EmbeddingModel, pitanje: str, k=5) -> list[Rezultat]:
    upit = model.embeduj_upite([pitanje]).tolist()
    odgovor = kolekcija.query(query_embeddings=upit, n_results=k)
    return [
        Rezultat(izvor=metapodaci["izvor"], tekst=tekst, distanca=distanca)
        for metapodaci, tekst, distanca in zip(
            odgovor["metadatas"][0],
            odgovor["documents"][0],
            odgovor["distances"][0],
            strict=True,
        )
    ]
