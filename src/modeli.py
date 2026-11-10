"""Embedding modeli koji se porede u eksperimentima."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpisModela:
    ime: str
    kratko_ime: str
    opis: str
    dimenzija: int
    prefiks_upita: str = ""
    prefiks_dokumenta: str = ""


BGE_M3 = OpisModela("BAAI/bge-m3", "bge-m3", "multiling (velik)", 1024)

MODELI = [
    OpisModela("all-MiniLM-L6-v2", "MiniLM-L6 (engleski)", "engleski (mali)", 384),
    OpisModela(
        "paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12 (višejezični)", "multiling (mali)", 384
    ),
    # e5 modeli su trenirani sa ovim prefiksima i bez njih daju lošije rezultate.
    OpisModela(
        "intfloat/multilingual-e5-large",
        "multilingual-e5-large",
        "multiling (velik)",
        1024,
        prefiks_upita="query: ",
        prefiks_dokumenta="passage: ",
    ),
    BGE_M3,
]

NAJBOLJI_MODEL = BGE_M3
