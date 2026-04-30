"""
visemes.py — Phoneme → Viseme mapping for Edlio Presence Layer.

Maps ARPAbet phoneme symbols (as output by faster-whisper word-level alignment)
to a standard set of ~15 viseme classes suitable for driving mouth shapes in
MuseTalk, Wav2Lip, or any compatible face renderer.

Viseme set used: Preston Blair / Disney Animation standard (industry baseline).
These 15 visemes cover all English phonemes and are widely used in game engines,
virtual avatars, and lip-sync systems.

References:
  - Preston Blair (1994) "Cartoon Animation" — classic viseme chart
  - OVR Lip Sync SDK viseme definitions
  - Blend shape conventions in MuseTalk / standard avatar rigs
"""

from typing import Optional

# ─── Viseme definitions ────────────────────────────────────────────────────────
#
#  ID  | Name      | Description                   | Example phonemes
# ─────┼───────────┼───────────────────────────────┼────────────────────────────
#   0  | sil       | Silence / neutral              | <silence>
#   1  | PP        | Bilabial plosive (lips closed) | P, B, M
#   2  | FF        | Labiodental fricative          | F, V
#   3  | TH        | Dental fricative               | TH (θ, ð)
#   4  | DD        | Alveolar stop                  | T, D
#   5  | kk        | Velar stop                     | K, G, NG
#   6  | CH        | Postalveolar affricate         | CH, JH, SH, ZH
#   7  | SS        | Alveolar sibilant              | S, Z
#   8  | nn        | Alveolar nasal                 | N, L
#   9  | RR        | Rhotic                         | R, ER
#  10  | aa        | Open front vowel               | AA, AH (unstressed)
#  11  | E         | Mid-front vowel                | EH, AE
#  12  | ih        | Near-close front vowel         | IH, IY
#  13  | oh        | Mid-back rounded vowel         | AO, AW, OY
#  14  | ou        | Close back rounded vowel       | OW, UH, UW
#  15  | W         | Labial-velar approximant       | W, OW (onset)

VISEME_NAMES = [
    "sil",   # 0
    "PP",    # 1
    "FF",    # 2
    "TH",    # 3
    "DD",    # 4
    "kk",    # 5
    "CH",    # 6
    "SS",    # 7
    "nn",    # 8
    "RR",    # 9
    "aa",    # 10
    "E",     # 11
    "ih",    # 12
    "oh",    # 13
    "ou",    # 14
    "W",     # 15
]

NUM_VISEMES = len(VISEME_NAMES)  # 16

# ARPAbet → Viseme mapping
# ARPAbet is what faster-whisper produces for English.
ARPABET_TO_VISEME: dict[str, int] = {
    # Silence / filler
    "SIL": 0, "SP": 0, "SPN": 0, "<sil>": 0, "": 0,

    # Bilabials (lips pressed together)
    "P": 1, "B": 1, "M": 1,

    # Labiodentals
    "F": 2, "V": 2,

    # Dentals
    "TH": 3, "DH": 3,

    # Alveolar stops
    "T": 4, "D": 4,

    # Velars
    "K": 5, "G": 5, "NG": 5,

    # Postalveolars / Affricates
    "CH": 6, "JH": 6, "SH": 6, "ZH": 6,

    # Sibilants
    "S": 7, "Z": 7,

    # Alveolar nasals and laterals
    "N": 8, "L": 8,

    # Rhotics
    "R": 9, "ER": 9, "AXR": 9,

    # Open vowels
    "AA": 10, "AH": 10, "AX": 10,

    # Mid-front vowels
    "EH": 11, "AE": 11,

    # Close-front vowels
    "IH": 12, "IY": 12, "IX": 12,

    # Mid-back rounded vowels
    "AO": 13, "AW": 13, "OY": 13,

    # Close-back rounded vowels
    "OW": 14, "UH": 14, "UW": 14, "UX": 14,

    # Labial-velar glide
    "W": 15, "HH": 0,  # H is near-silent; treat as neutral

    # Palatal glide — maps to close-front (IY-like shape)
    "Y": 12,
}

# IPA character → Viseme mapping (for pipelines that output IPA)
# Subset of common IPA symbols used by English TTS / phonemizers.
IPA_TO_VISEME: dict[str, int] = {
    # Silence
    " ": 0, "": 0,

    # Bilabials
    "p": 1, "b": 1, "m": 1,

    # Labiodentals
    "f": 2, "v": 2,

    # Dentals
    "θ": 3, "ð": 3,

    # Alveolars
    "t": 4, "d": 4,
    "n": 8, "l": 8,
    "s": 7, "z": 7,

    # Velars
    "k": 5, "ɡ": 5, "g": 5, "ŋ": 5,

    # Postalveolars
    "ʃ": 6, "ʒ": 6, "tʃ": 6, "dʒ": 6, "t͡ʃ": 6, "d͡ʒ": 6,

    # Rhotics
    "r": 9, "ɹ": 9, "ɚ": 9, "ɝ": 9,

    # Glides
    "w": 15, "j": 12, "h": 0,

    # Vowels — open
    "ɑ": 10, "a": 10, "ʌ": 10, "ə": 10,

    # Vowels — mid-front
    "ɛ": 11, "æ": 11, "e": 11,

    # Vowels — close-front
    "ɪ": 12, "i": 12, "iː": 12,

    # Vowels — mid-back
    "ɔ": 13, "ɑʊ": 13, "aʊ": 13, "ɔɪ": 13,

    # Vowels — close-back
    "oʊ": 14, "o": 14, "ʊ": 14, "u": 14, "uː": 14,
}


def arpabet_to_viseme(phoneme: str) -> int:
    """
    Convert an ARPAbet phoneme string to a viseme ID (0–15).
    Strips trailing stress markers (0/1/2) before lookup.
    Returns 0 (silence) if the phoneme is not recognized.
    """
    # Strip stress digits: "AH1" → "AH", "IY0" → "IY"
    stripped = phoneme.upper().rstrip("012")
    return ARPABET_TO_VISEME.get(stripped, 0)


def ipa_to_viseme(phoneme: str) -> int:
    """
    Convert an IPA phoneme symbol to a viseme ID (0–15).
    Returns 0 (silence) if the phoneme is not recognized.
    """
    # Try exact match first, then single-char lookup
    v = IPA_TO_VISEME.get(phoneme)
    if v is not None:
        return v
    # Try first character for compound symbols
    if len(phoneme) > 1:
        return IPA_TO_VISEME.get(phoneme[0], 0)
    return 0


def viseme_name(viseme_id: int) -> str:
    """Return the human-readable name for a viseme ID."""
    if 0 <= viseme_id < NUM_VISEMES:
        return VISEME_NAMES[viseme_id]
    return "unknown"


if __name__ == "__main__":
    # Quick sanity check
    test_cases = [
        ("P", "ARPAbet bilabial plosive"),
        ("AH1", "ARPAbet unstressed vowel with stress digit"),
        ("TH", "ARPAbet dental"),
        ("NG", "ARPAbet velar nasal"),
        ("p", "IPA bilabial"),
        ("ə", "IPA schwa"),
        ("ŋ", "IPA velar nasal"),
        ("UNKNOWN_XYZ", "unknown phoneme → silence"),
    ]
    print("Phoneme → Viseme mapping sanity check:")
    print(f"{'Phoneme':<20} {'Description':<35} {'Viseme ID':<12} {'Viseme Name'}")
    print("─" * 80)
    for phoneme, desc in test_cases:
        # Always try ARPAbet first (handles stress digits via strip); fall back to IPA
        arp_result = arpabet_to_viseme(phoneme)
        # If ARPAbet returned 0 (unknown/silence), also try IPA
        if arp_result == 0 and phoneme not in ARPABET_TO_VISEME and phoneme.upper().rstrip('012') not in ARPABET_TO_VISEME:
            vid = ipa_to_viseme(phoneme)
        else:
            vid = arp_result
        print(f"{phoneme:<20} {desc:<35} {vid:<12} {viseme_name(vid)}")
