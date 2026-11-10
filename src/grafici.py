"""
Grafici i tabele za rad (PNG), napravljeni iz rezultata u rezultati/raw/.

Tabele nemaju naslov u slici; naslov i tumačenje idu u tekst rada.
Svaki model ima istu boju na svim graficima.

Pokretanje iz glavnog foldera: python src/grafici.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from chunking import OVERLAPI, SECENJA, VELICINE  # noqa: E402
from modeli import MODELI  # noqa: E402
from rezultati import FOLDER_SLIKA, ucitaj_csv  # noqa: E402

# Redosled prikaza (najbolji gore/levo) i boja koja uvek prati isti model.
REDOSLED_MODELA = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
]
BOJA_MODELA = {
    "BAAI/bge-m3": "#2a78d6",
    "intfloat/multilingual-e5-large": "#eb6834",
    "paraphrase-multilingual-MiniLM-L12-v2": "#1baf7a",
    "all-MiniLM-L6-v2": "#eda100",
}
KRATKO_IME = {m.ime: m.kratko_ime for m in MODELI}
ENGLESKI_MODEL = "all-MiniLM-L6-v2"

NAZIV_SECENJA = {"standardno": "Standardno", "sentence-aware": "Sentence-aware"}
BOJA_SECENJA = {"standardno": "#9db1c9", "sentence-aware": "#2a78d6"}

TEKST = "#0b0b0b"
TEKST_2 = "#52514e"
MREZA = "#e4e3df"
ZAGLAVLJE_TABELE = "#1e293b"
POZADINA_ISTAKNUTOG = "#dbeafe"
TEKST_ISTAKNUTOG = "#104281"
POZADINA_PARNOG_REDA = "#f8fafc"
IVICA_TABELE = "#e2e8f0"

PLAVA_SKALA = LinearSegmentedColormap.from_list(
    "plava",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": TEKST,
        "axes.labelcolor": TEKST_2,
        "xtick.color": TEKST_2,
        "ytick.color": TEKST_2,
        "axes.edgecolor": "#c9c8c3",
        "axes.linewidth": 0.8,
    }
)


def napravi_sve():
    print("Pravim grafike i tabele:")
    rq1 = _ucitaj_rq1()
    if rq1:
        grafik_rq1_toplotna_mapa(rq1)
        grafik_rq1_velicina_chunka(rq1)
        grafik_rq1_nacin_secenja(rq1)
        grafik_rq1_kvalitet_latencija(rq1)
        tabela_rq1(rq1)
    tabela_rq2()
    tabela_rq3()


def grafik_rq1_toplotna_mapa(redovi):
    kombinacije = [(s, v, o) for s in SECENJA for v in VELICINE for o in OVERLAPI]
    matrica = np.array([[_mrr(redovi, m, *k) for k in kombinacije] for m in REDOSLED_MODELA])

    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    slika = ax.imshow(matrica, cmap=PLAVA_SKALA, vmin=0, vmax=1, aspect="auto")
    _upisi_vrednosti_u_polja(ax, matrica)
    _razdvoji_grupe(ax, len(kombinacije))

    ax.set_xticks(range(len(kombinacije)))
    ax.set_xticklabels([f"{o}%" for _, _, o in kombinacije], fontsize=8)
    ax.set_yticks(range(len(REDOSLED_MODELA)))
    ax.set_yticklabels([KRATKO_IME[m] for m in REDOSLED_MODELA], fontsize=9.5)
    ax.tick_params(length=0)
    for ivica in ax.spines.values():
        ivica.set_visible(False)
    _oznaci_velicine_i_secenja(ax)

    traka = fig.colorbar(slika, ax=ax, fraction=0.025, pad=0.015)
    traka.set_label("MRR", color=TEKST_2)
    traka.outline.set_visible(False)
    traka.ax.tick_params(color=TEKST_2, labelsize=8)

    ax.set_title(
        "MRR za svaki model i svaku kombinaciju chunkinga (veličina / preklapanje)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    _sacuvaj(fig, "rq1_toplotna_mapa.png")


def grafik_rq1_velicina_chunka(redovi):
    fig, osi = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    x = np.arange(len(VELICINE))

    for ax, secenje in zip(osi, SECENJA, strict=True):
        krajevi = []
        for model in REDOSLED_MODELA:
            y = [np.mean([_mrr(redovi, model, secenje, v, o) for o in OVERLAPI]) for v in VELICINE]
            ax.plot(
                x,
                y,
                color=BOJA_MODELA[model],
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=KRATKO_IME[model],
                zorder=3,
            )
            krajevi.append(y[-1])
        _oznaci_krajeve_linija(ax, x[-1], krajevi)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v} tokena" for v in VELICINE])
        ax.set_xlim(-0.25, len(VELICINE) - 0.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"{NAZIV_SECENJA[secenje]} sečenje", fontsize=10.5, fontweight="bold")
        _ocisti_ose(ax)

    osi[0].set_ylabel("MRR (prosek preko preklapanja 0/10/20%)")
    linije, nazivi = osi[0].get_legend_handles_labels()
    fig.legend(
        linije,
        nazivi,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9.5,
    )
    fig.suptitle(
        "Uticaj veličine chunka na MRR zavisi od modela", fontsize=11.5, fontweight="bold", y=1.0
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _sacuvaj(fig, "rq1_velicina_chunka.png")


def grafik_rq1_nacin_secenja(redovi):
    # Engleski model je izostavljen: na srpskom je oko nule u svim kombinacijama.
    modeli = [m for m in REDOSLED_MODELA if m != ENGLESKI_MODEL]
    x = np.arange(len(modeli))
    sirina = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, secenje in enumerate(SECENJA):
        proseci = [
            np.mean([r["MRR"] for r in redovi if r["model"] == m and r["secenje"] == secenje])
            for m in modeli
        ]
        stubici = ax.bar(
            x + (k - 0.5) * (sirina + 0.02),
            proseci,
            sirina,
            color=BOJA_SECENJA[secenje],
            label=f"{NAZIV_SECENJA[secenje]} sečenje",
            zorder=3,
        )
        for stubic, vrednost in zip(stubici, proseci, strict=True):
            ax.text(
                stubic.get_x() + stubic.get_width() / 2,
                vrednost + 0.012,
                f"{vrednost:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEKST_2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([KRATKO_IME[m] for m in modeli])
    ax.set_ylabel("MRR (prosek preko svih 9 kombinacija)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.set_title(
        "Standardno naspram sentence-aware sečenja", fontsize=11.5, fontweight="bold", pad=10
    )
    _ocisti_ose(ax)
    _sacuvaj(fig, "rq1_nacin_secenja.png")


def grafik_rq1_kvalitet_latencija(redovi):
    # Dva velika modela su blizu, pa im natpisi idu na suprotne strane tačke.
    pomeraj_natpisa = {
        "BAAI/bge-m3": (-10, 18, "right"),
        "intfloat/multilingual-e5-large": (10, -20, "left"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for model in REDOSLED_MODELA:
        najbolji = _najbolji_red(redovi, model)
        tacka = (najbolji["latencija_ms"], najbolji["MRR"])
        ax.scatter(*tacka, s=90, color=BOJA_MODELA[model], edgecolor="white", linewidth=2, zorder=3)
        dx, dy, poravnanje = pomeraj_natpisa.get(model, (10, -2, "left"))
        ax.annotate(
            f"{KRATKO_IME[model]}\nMRR {najbolji['MRR']:.3f}, {najbolji['latencija_ms']:.0f} ms",
            tacka,
            xytext=(dx, dy),
            textcoords="offset points",
            va="center",
            ha=poravnanje,
            fontsize=8.5,
            color=TEKST,
        )

    ax.set_xscale("log")
    ax.set_xlim(4, 600)
    ax.set_ylim(-0.03, 1)
    podeoci = [5, 10, 20, 50, 100, 200]
    ax.set_xticks(podeoci)
    ax.set_xticklabels([str(p) for p in podeoci])
    ax.set_xlabel("Latencija upita u ms (embedovanje pitanja + pretraga), log skala")
    ax.set_ylabel("MRR (najbolja konfiguracija modela)")
    ax.set_title("Kvalitet naspram brzine", fontsize=11.5, fontweight="bold", pad=10)
    _ocisti_ose(ax)
    ax.grid(axis="x", color=MREZA, linewidth=0.8)
    _sacuvaj(fig, "rq1_kvalitet_latencija.png")


def tabela_rq1(redovi):
    modeli = sorted(REDOSLED_MODELA, key=lambda m: -_najbolji_red(redovi, m)["MRR"])
    kolone = [
        "Model",
        "Dim",
        "Najbolja konfiguracija",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "MRR",
        "Prosečan\nMRR",
        "Latencija\n(ms)",
        "Indeksiranje\n(chunk/s)",
    ]
    redovi_tabele = []
    for model in modeli:
        r = _najbolji_red(redovi, model)
        prosek = np.mean([x["MRR"] for x in redovi if x["model"] == model])
        redovi_tabele.append(
            [
                KRATKO_IME[model],
                f"{r['dimenzija']}",
                f"{r['secenje']}, {r['velicina']} / {r['overlap']}%",
                _procenat(r["hit_rate@1"]),
                _procenat(r["hit_rate@3"]),
                _procenat(r["hit_rate@5"]),
                f"{r['MRR']:.3f}",
                f"{prosek:.3f}",
                f"{r['latencija_ms']:.0f}",
                f"{r['throughput_chunk_s']:.1f}",
            ]
        )
    _tabela(
        redovi_tabele, kolone, "rq1_tabela.png", sirina=12.5, istaknut_red=0, levo_poravnate=(0, 2)
    )


def tabela_rq2():
    redovi = ucitaj_csv("rq2_metrike_udaljenosti.csv")
    if not redovi:
        print("  rq2_metrike_udaljenosti.csv ne postoji; prvo pokreni RQ2.")
        return
    kolone = [
        "Metrika",
        "Prosečna norma\nvektora",
        "Gradnja indeksa\n(ms)",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "MRR",
    ]
    redovi_tabele = [
        [
            r["metrika_naziv"],
            f"{r['prosecna_norma_vektora']:.4f}",
            f"{r['vreme_gradnje_indeksa_ms']:.0f}",
            _procenat(r["hit_rate@1"]),
            _procenat(r["hit_rate@3"]),
            _procenat(r["hit_rate@5"]),
            f"{r['MRR']:.4f}",
        ]
        for r in redovi
    ]
    _tabela(redovi_tabele, kolone, "rq2_tabela.png", sirina=9, levo_poravnate=(0,))


def tabela_rq3():
    redovi = ucitaj_csv("rq3_hnsw_parametri.csv")
    if not redovi:
        print("  rq3_hnsw_parametri.csv ne postoji; prvo pokreni RQ3.")
        return
    kolone = [
        "M",
        "ef_construction",
        "Gradnja indeksa\n(ms)",
        "Pretraga\n(ms)",
        "Recall@5 u odnosu\nna tačnu pretragu",
        "Hit@1",
        "Hit@5",
        "MRR",
    ]
    redovi_tabele = [
        [
            f"{r['M']}",
            f"{r['ef_construction']}",
            f"{r['vreme_gradnje_indeksa_ms']:.0f}",
            f"{r['latencija_pretrage_ms']:.2f}",
            f"{r['recall@5_vs_tacna']:.3f}",
            _procenat(r["hit_rate@1"]),
            _procenat(r["hit_rate@5"]),
            f"{r['MRR']:.4f}",
        ]
        for r in redovi
    ]
    _tabela(redovi_tabele, kolone, "rq3_tabela.png", sirina=10)


def _ucitaj_rq1():
    redovi = ucitaj_csv("rq1_modeli_chunking.csv")
    ocekivano = len(MODELI) * len(SECENJA) * len(VELICINE) * len(OVERLAPI)
    if len(redovi) < ocekivano:
        print(f"  RQ1 nije završen ({len(redovi)}/{ocekivano} merenja); preskačem RQ1 grafike.")
        return None
    return redovi


def _mrr(redovi, model, secenje, velicina, overlap) -> float:
    return next(
        r["MRR"]
        for r in redovi
        if r["model"] == model
        and r["secenje"] == secenje
        and r["velicina"] == velicina
        and r["overlap"] == overlap
    )


def _najbolji_red(redovi, model) -> dict:
    return max((r for r in redovi if r["model"] == model), key=lambda r: r["MRR"])


def _procenat(udeo: float) -> str:
    return f"{udeo * 100:.1f}%"


def _upisi_vrednosti_u_polja(ax, matrica):
    najbolje = np.unravel_index(np.argmax(matrica), matrica.shape)
    for (i, j), vrednost in np.ndenumerate(matrica):
        ax.text(
            j,
            i,
            f"{vrednost:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if vrednost > 0.55 else TEKST,
            fontweight="bold" if (i, j) == najbolje else "normal",
        )


def _razdvoji_grupe(ax, broj_kolona):
    po_velicini = len(OVERLAPI)
    po_secenju = len(VELICINE) * po_velicini
    for j in range(po_velicini, broj_kolona, po_velicini):
        ax.axvline(j - 0.5, color="white", linewidth=3.5 if j % po_secenju == 0 else 1.5)
    for i in range(1, len(REDOSLED_MODELA)):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)


def _oznaci_velicine_i_secenja(ax):
    dno = len(REDOSLED_MODELA) - 0.5
    po_velicini = len(OVERLAPI)
    po_secenju = len(VELICINE) * po_velicini
    for g, secenje in enumerate(SECENJA):
        for k, velicina in enumerate(VELICINE):
            sredina = g * po_secenju + k * po_velicini + (po_velicini - 1) / 2
            ax.text(
                sredina,
                dno + 0.55,
                f"{velicina} tokena",
                ha="center",
                va="top",
                fontsize=8.5,
                color=TEKST_2,
            )
        ax.text(
            g * po_secenju + (po_secenju - 1) / 2,
            dno + 0.95,
            f"{NAZIV_SECENJA[secenje]} sečenje",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=TEKST,
        )


def _oznaci_krajeve_linija(ax, x_kraja, vrednosti):
    # Bliske vrednosti se razmiču vertikalno da se natpisi ne preklapaju.
    razmak = 0.045
    pozicije = {}
    prethodna = -1.0
    for i in sorted(range(len(vrednosti)), key=lambda i: vrednosti[i]):
        pozicije[i] = max(vrednosti[i], prethodna + razmak)
        prethodna = pozicije[i]
    for i, vrednost in enumerate(vrednosti):
        ax.annotate(
            f"{vrednost:.2f}",
            (x_kraja, vrednost),
            xytext=(x_kraja + 0.08, pozicije[i]),
            textcoords="data",
            va="center",
            fontsize=8.5,
            color=TEKST_2,
        )


def _ocisti_ose(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=MREZA, linewidth=0.8)
    ax.set_axisbelow(True)


def _tabela(redovi, kolone, ime_fajla, *, sirina, istaknut_red=None, levo_poravnate=()):
    visina = 0.45 + 0.36 * (len(redovi) + 1)
    fig, ax = plt.subplots(figsize=(sirina, visina))
    ax.axis("off")
    tabela = ax.table(cellText=redovi, colLabels=kolone, loc="center", cellLoc="center")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 1.6)
    tabela.auto_set_column_width(list(range(len(kolone))))

    zaglavlje_u_dva_reda = any("\n" in k for k in kolone)
    for (red, kolona), polje in tabela.get_celld().items():
        polje.set_edgecolor(IVICA_TABELE)
        polje.PAD = 0.06
        if red == 0:
            if zaglavlje_u_dva_reda:
                polje.set_height(polje.get_height() * 1.7)
            polje.set_facecolor(ZAGLAVLJE_TABELE)
            polje.set_text_props(color="white", fontweight="bold")
        elif istaknut_red is not None and red - 1 == istaknut_red:
            polje.set_facecolor(POZADINA_ISTAKNUTOG)
            polje.set_text_props(fontweight="bold", color=TEKST_ISTAKNUTOG)
        elif red % 2 == 0:
            polje.set_facecolor(POZADINA_PARNOG_REDA)
        if kolona in levo_poravnate and red > 0:
            polje.set_text_props(ha="left")
            polje.PAD = 0.03
    _sacuvaj(fig, ime_fajla)


def _sacuvaj(fig, ime_fajla):
    os.makedirs(FOLDER_SLIKA, exist_ok=True)
    putanja = os.path.join(FOLDER_SLIKA, ime_fajla)
    fig.savefig(putanja, bbox_inches="tight", facecolor="white", pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"  {os.path.relpath(putanja)}")


if __name__ == "__main__":
    napravi_sve()
