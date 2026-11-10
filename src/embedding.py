"""Omotač oko SentenceTransformer modela koji dodaje prefikse koje model očekuje."""

import numpy as np
from sentence_transformers import SentenceTransformer

from modeli import OpisModela


class EmbeddingModel:
    def __init__(self, opis: OpisModela):
        self.opis = opis
        self._model = SentenceTransformer(opis.ime)

    def embeduj_dokumente(self, tekstovi: list[str]) -> np.ndarray:
        return self._embeduj(self.opis.prefiks_dokumenta, tekstovi)

    def embeduj_upite(self, upiti: list[str]) -> np.ndarray:
        return self._embeduj(self.opis.prefiks_upita, upiti)

    def embeduj_upit(self, upit: str) -> np.ndarray:
        return self.embeduj_upite([upit])[0]

    def _embeduj(self, prefiks: str, tekstovi: list[str]) -> np.ndarray:
        return self._model.encode([prefiks + t for t in tekstovi], show_progress_bar=False)
