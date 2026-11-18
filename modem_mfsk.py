#!/usr/bin/env python3
"""
Modem audio M-FSK : envoyer un texte d'un PC à un autre par le son.
Emilien Henskens · Cian Jones · Daniel Pinto Guimaraes Amaral

Paramètres (cf. état des lieux) :
    fs = 44,1 kHz · Ts = 100 ms · f0 = 1000 Hz · Δf = 100 Hz
    préambule : chirp montant 500 → 5000 Hz
    M ∈ {2, 4, 8, 16, 32}   →   débit = log2(M) / Ts

Trame émise :
    [préambule][silence][en-tête : longueur du message sur 8 bits][message UTF-8]

Utilisation :
    python modem_mfsk.py emit "Bonjour" -M 4 -o bonjour.wav        # crée le .wav
    python modem_mfsk.py emit "Bonjour" -M 4 --play               # joue directement
    python modem_mfsk.py recv bonjour.wav -M 4                     # décode un .wav
    python modem_mfsk.py recv --record 8 -M 4                      # enregistre 8 s au micro puis décode
    python modem_mfsk.py test                                      # boucle sans bruit sur les 5 textes × 5 M
    python modem_mfsk.py figures -o figures/                       # figures du poster (PNG 300 dpi)

Dépendances : numpy, scipy, matplotlib ; sounddevice (optionnel, pour --play / --record).
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate

# --------------------------------------------------------------------------- #
# Paramètres
# --------------------------------------------------------------------------- #
FS = 44_100          # fréquence d'échantillonnage (Hz)
TS = 0.100           # durée d'une note (s)
F0 = 1000.0          # première note (Hz)
DF = 100.0           # écart entre notes (Hz)
PRE_F = (500.0, 5000.0)
PRE_DUR = 0.5        # durée du préambule (s)
GAP_DUR = 0.1        # silence entre préambule et données (s)
FADE = 0.005         # rampe d'entrée/sortie de chaque note (s), évite les clics
AMP = 0.8
HEADER_BITS = 8      # longueur du message (en octets) codée sur 8 bits → ≤ 255 octets

VALENCES = (2, 4, 8, 16, 32)

TEXTES_TEST = {
    "T1": "Bonjour",
    "T2": "Portez ce vieux whisky au juge blond qui fume",
    "T3": "0123456789 +-*/=()[]{}#@&%$!?",
    "T4": "Élève, ça déçoit à Noël : où est l'âme du maïs ?",
    "T5": "Le modem audio transmet ce texte d'un ordinateur à un autre, uniquement grâce au son émis.",
}

# Palette du poster
BLEU, ORANGE, NUIT, GRIS, FOND = "#0072B2", "#E69F00", "#1B2A41", "#5A6270", "#F7F7F2"


def bits_par_symbole(M: int) -> int:
    k = int(np.log2(M))
    if 2 ** k != M or M < 2:
        raise ValueError(f"M doit être une puissance de 2 (≥ 2), reçu {M}")
    return k


def frequences(M: int) -> np.ndarray:
    return F0 + DF * np.arange(M)


def debit(M: int) -> float:
    return bits_par_symbole(M) / TS


# --------------------------------------------------------------------------- #
# Texte ↔ bits ↔ symboles
# --------------------------------------------------------------------------- #
def texte_vers_bits(texte: str) -> np.ndarray:
    data = texte.encode("utf-8")
    if len(data) > 2 ** HEADER_BITS - 1:
        raise ValueError(f"Message trop long ({len(data)} octets, max {2 ** HEADER_BITS - 1})")
    octets = bytes([len(data)]) + data
    return np.unpackbits(np.frombuffer(octets, dtype=np.uint8))


def bits_vers_symboles(bits: np.ndarray, M: int) -> np.ndarray:
    k = bits_par_symbole(M)
    pad = (-len(bits)) % k
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    groupes = bits.reshape(-1, k)
    poids = 2 ** np.arange(k - 1, -1, -1)
    return groupes @ poids


def symboles_vers_bits(symboles: np.ndarray, M: int) -> np.ndarray:
    k = bits_par_symbole(M)
    return ((symboles[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def bits_vers_texte(bits: np.ndarray) -> str:
    n = int(np.packbits(bits[:HEADER_BITS])[0])
    corps = bits[HEADER_BITS:HEADER_BITS + 8 * n]
    corps = corps[: len(corps) - len(corps) % 8]
    return np.packbits(corps).tobytes().decode("utf-8", errors="replace")


def nb_symboles(n_octets: int, M: int) -> int:
    k = bits_par_symbole(M)
    return int(np.ceil((HEADER_BITS + 8 * n_octets) / k))


# --------------------------------------------------------------------------- #
# Émetteur
# --------------------------------------------------------------------------- #
def preambule() -> np.ndarray:
    t = np.arange(int(PRE_DUR * FS)) / FS
    return AMP * chirp(t, PRE_F[0], PRE_DUR, PRE_F[1], method="linear") * _fenetre_fade(len(t))


def _fenetre_fade(n: int) -> np.ndarray:
    nf = int(FADE * FS)
    w = np.ones(n)
    rampe = 0.5 * (1 - np.cos(np.pi * np.arange(nf) / nf))
    w[:nf], w[-nf:] = rampe, rampe[::-1]
    return w


def moduler(texte: str, M: int) -> np.ndarray:
    """Texte → signal audio (float, [-1, 1])."""
    symboles = bits_vers_symboles(texte_vers_bits(texte), M)
    ns = int(TS * FS)
    t = np.arange(ns) / FS
    fen = _fenetre_fade(ns)
    freqs = frequences(M)
    notes = [AMP * np.sin(2 * np.pi * freqs[s] * t) * fen for s in symboles]
    silence = np.zeros(int(GAP_DUR * FS))
    return np.concatenate([silence, preambule(), silence, *notes, silence])


# --------------------------------------------------------------------------- #
# Récepteur
# --------------------------------------------------------------------------- #
def detecter_debut(x: np.ndarray) -> int:
    """Indice du premier échantillon de données (après préambule + silence)."""
    ref = preambule()
    c = np.abs(correlate(x, ref, mode="valid", method="fft"))
    return int(np.argmax(c)) + len(ref) + int(GAP_DUR * FS)


def energie_par_note(tranche: np.ndarray, M: int) -> np.ndarray:
    """|X(f)| aux M fréquences de la constellation (DFT ciblée, fenêtre de Hann)."""
    n = len(tranche)
    t = np.arange(n) / FS
    w = np.hanning(n)
    base = np.exp(-2j * np.pi * np.outer(frequences(M), t))
    return np.abs(base @ (tranche * w))


def demoduler(x: np.ndarray, M: int, details: bool = False):
    """Signal audio → texte. Lit d'abord l'en-tête pour savoir combien de notes décoder."""
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= np.max(np.abs(x)) + 1e-12

    debut = detecter_debut(x)
    ns = int(TS * FS)
    marge = int(0.1 * ns)  # on ignore 10 % de chaque bord (transitoires, échos)

    def symbole(i):
        a = debut + i * ns + marge
        b = debut + (i + 1) * ns - marge
        if b > len(x):
            return 0
        return int(np.argmax(energie_par_note(x[a:b], M)))

    k = bits_par_symbole(M)
    n_head = int(np.ceil(HEADER_BITS / k))
    head = np.array([symbole(i) for i in range(n_head)])
    n_octets = int(np.packbits(symboles_vers_bits(head, M)[:HEADER_BITS])[0])
    n_sym = nb_symboles(n_octets, M)
    symboles = np.array([symbole(i) for i in range(n_sym)])
    bits = symboles_vers_bits(symboles, M)
    texte = bits_vers_texte(bits)
    if details:
        return texte, bits, symboles, debut
    return texte


# --------------------------------------------------------------------------- #
# Mesures
# --------------------------------------------------------------------------- #
def teb(texte_envoye: str, bits_recus: np.ndarray) -> float:
    """Taux d'erreur binaire = bits faux / bits envoyés (en-tête compris)."""
    ref = texte_vers_bits(texte_envoye)
    rec = bits_recus[: len(ref)]
    if len(rec) < len(ref):
        rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.uint8)])
    return float(np.mean(ref != rec))


def ajouter_bruit(x: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    p = np.mean(x[np.abs(x) > 0] ** 2)
    sigma = np.sqrt(p / 10 ** (snr_db / 10))
    return x + rng.normal(0, sigma, len(x))


# --------------------------------------------------------------------------- #
# Audio (optionnel)
# --------------------------------------------------------------------------- #
def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as e:  # PortAudio absent, etc.
        sys.exit(f"sounddevice indisponible ({e}). Installe-le : pip install sounddevice")


def ecrire_wav(chemin: str, x: np.ndarray):
    wavfile.write(chemin, FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def lire_wav(chemin: str) -> np.ndarray:
    fs, x = wavfile.read(chemin)
    if fs != FS:
        from scipy.signal import resample_poly
        x = resample_poly(x.astype(float), FS, fs, axis=0)
    return x.astype(float)


# --------------------------------------------------------------------------- #
# Figures du poster
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.edgecolor": GRIS, "axes.labelcolor": NUIT, "xtick.color": GRIS, "ytick.color": GRIS,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    })
    return plt


def figures(dossier: str, essais: int = 20):
    plt = _style()
    os.makedirs(dossier, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. Débit selon M
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = [debit(M) for M in VALENCES]
    ax.bar([str(M) for M in VALENCES], d, color=BLEU, width=0.6)
    for i, v in enumerate(d):
        ax.text(i, v + 1, f"{v:.0f} b/s", ha="center", color=NUIT, fontweight="bold")
    ax.set_xlabel("Valence M"); ax.set_ylabel("Débit (bits/s)")
    ax.set_ylim(0, max(d) * 1.2)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "debit_vs_M.png")); plt.close(fig)

    # 2. La lettre « B » avec M = 4 : notes émises
    M = 4
    sym = bits_vers_symboles(np.unpackbits(np.frombuffer(b"B", np.uint8)), M)
    fig, axes = plt.subplots(2, len(sym), figsize=(12, 5), sharey="row")
    t = np.arange(int(0.01 * FS)) / FS  # 10 ms visibles
    for i, s in enumerate(sym):
        f = frequences(M)[s]
        axes[0, i].plot(t * 1000, np.sin(2 * np.pi * f * t), color=BLEU, lw=2)
        bits = format(s, "02b")
        axes[0, i].set_title(f"{bits} → {f:.0f} Hz", color=NUIT, fontweight="bold")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        tranche = np.sin(2 * np.pi * f * np.arange(int(TS * FS)) / FS)
        spec = np.abs(np.fft.rfft(tranche * np.hanning(len(tranche))))
        fr = np.fft.rfftfreq(len(tranche), 1 / FS)
        m = (fr > 800) & (fr < 1500)
        axes[1, i].plot(fr[m], spec[m] / spec.max(), color=ORANGE, lw=2)
        axes[1, i].set_xlabel("Hz"); axes[1, i].set_yticks([])
    axes[1, 0].set_ylabel("FFT")
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "lettre_B_M4.png")); plt.close(fig)

    # 3. Spectrogramme d'un message émis
    x = moduler("Bonjour", 4) + 1e-6 * rng.normal(size=len(moduler("Bonjour", 4)))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.specgram(x, NFFT=2048, Fs=FS, noverlap=1536, cmap="Blues")
    ax.set_ylim(0, 5500); ax.set_xlabel("Temps (s)"); ax.set_ylabel("Fréquence (Hz)")
    ax.annotate("préambule", xy=(0.35, 4500), color=NUIT, fontweight="bold")
    ax.annotate("« Bonjour » (M = 4)", xy=(0.9, 1600), color=NUIT, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "spectrogramme_bonjour.png")); plt.close(fig)

    # 4. TEB selon le SNR (simulation avec bruit), une courbe par M
    snrs = np.arange(-30, -9, 2)
    couleurs = [BLEU, ORANGE, "#009E73", "#CC79A7", NUIT]
    fig, ax = plt.subplots(figsize=(8, 5))
    texte = TEXTES_TEST["T2"]
    for M, c in zip(VALENCES, couleurs):
        x = moduler(texte, M)
        res = []
        for snr in snrs:
            e = [teb(texte, demoduler(ajouter_bruit(x, snr, rng), M, details=True)[1]) for _ in range(essais)]
            res.append(max(np.mean(e), 1e-4))
        ax.semilogy(snrs, res, "o-", color=c, lw=2, label=f"M = {M}")
        print(f"  M = {M:2d} : TEB = " + " ".join(f"{r:.0e}" for r in res))
    ax.set_xlabel("Rapport signal sur bruit (dB)"); ax.set_ylabel("Taux d'erreur binaire")
    ax.set_ylim(1e-4, 1); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(dossier, "teb_vs_snr.png")); plt.close(fig)
    print(f"Figures enregistrées dans {dossier}/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Modem audio M-FSK")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="texte → son")
    e.add_argument("texte"); e.add_argument("-M", type=int, default=4)
    e.add_argument("-o", "--out", default="message.wav"); e.add_argument("--play", action="store_true")

    r = sub.add_parser("recv", help="son → texte")
    r.add_argument("wav", nargs="?"); r.add_argument("-M", type=int, default=4)
    r.add_argument("--record", type=float, metavar="SECONDES", help="enregistrer au micro")
    r.add_argument("--ref", help="texte envoyé, pour calculer le TEB")

    sub.add_parser("test", help="boucle sans bruit : 5 textes × 5 valeurs de M")

    f = sub.add_parser("figures", help="figures du poster")
    f.add_argument("-o", "--out", default="figures"); f.add_argument("--essais", type=int, default=20)

    a = p.parse_args()

    if a.cmd == "emit":
        x = moduler(a.texte, a.M)
        ecrire_wav(a.out, x)
        print(f"{a.out} : {len(x) / FS:.1f} s, M = {a.M}, débit = {debit(a.M):.0f} b/s")
        if a.play:
            sd = _sd(); sd.play(x, FS); sd.wait()

    elif a.cmd == "recv":
        if a.record:
            sd = _sd()
            print(f"Enregistrement {a.record} s…")
            x = sd.rec(int(a.record * FS), samplerate=FS, channels=1); sd.wait(); x = x[:, 0]
            ecrire_wav("enregistrement.wav", x / (np.max(np.abs(x)) + 1e-12))
        elif a.wav:
            x = lire_wav(a.wav)
        else:
            sys.exit("Donne un fichier .wav ou --record SECONDES")
        texte, bits, _, _ = demoduler(x, a.M, details=True)
        print(f"Texte reçu : {texte}")
        if a.ref:
            print(f"TEB : {teb(a.ref, bits):.4f}  ·  message exact : {texte == a.ref}")

    elif a.cmd == "test":
        ok = 0
        for nom, texte in TEXTES_TEST.items():
            for M in VALENCES:
                x = moduler(texte, M)
                recu, bits, _, _ = demoduler(x, M, details=True)
                bon = recu == texte; ok += bon
                print(f"{nom}  M={M:2d}  {len(x) / FS:5.1f} s  TEB={teb(texte, bits):.3f}  {'OK' if bon else 'ERREUR'}")
        print(f"\n{ok}/{len(TEXTES_TEST) * len(VALENCES)} messages décodés sans erreur")

    elif a.cmd == "figures":
        figures(a.out, a.essais)


if __name__ == "__main__":
    main()
