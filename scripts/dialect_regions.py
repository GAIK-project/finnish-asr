"""Dialect regions for SKN and LA-murre segments, and the region prompt a fine-tune can be given.

Six regions, the top level of LA-murre's own metadata (the `dialect_region` attribute of its VRT
files, http://urn.fi/urn:nbn:fi:lb-2019092001): LOU lounaismurteet, LVÄ lounaiset välimurteet, HÄM
hämäläismurteet, POH pohjalaismurteet (with Peräpohjola), SAV savolaismurteet, KAA kaakkoismurteet.

SKN: the dialect group in each recording's IMDI metadata (Metadata/SKN<nn>a.imdi, the Finnish
Description), mapped to the six: Etelä-, Keski- and Pohjois-Pohjanmaan murteet and Peräpohja are POH.
LA-murre: `lumi.sh prepare lamurre-regions` writes $FIASR_DATA/lamurre_meta/regions.tsv
(filename, dialect_region, dialect_group, parish) from the public VRT package; a segment id starts
with its recording's filename.

The prompt is Whisper's previous-text context, `<|startofprev|> Murre: savolaismurteet.`, before
<|startoftranscript|>. WRONG is the control: every region swapped with a far one, so a prompt that
only says "this is dialect" helps the right and the wrong region alike.
Standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))

NAMES = {
    "LOU": "lounaismurteet",
    "LVÄ": "lounaiset välimurteet",
    "HÄM": "hämäläismurteet",
    "POH": "pohjalaismurteet",
    "SAV": "savolaismurteet",
    "KAA": "kaakkoismurteet",
}
WRONG = {"LOU": "KAA", "LVÄ": "POH", "HÄM": "SAV", "POH": "LOU", "SAV": "LVÄ", "KAA": "HÄM"}

# SKN municipality -> region, from each recording's IMDI Description; the comments give the group named there
SKN = {
    "Velkua": "LOU",
    "Hinnerjoki": "LOU",
    "Eurajoki": "LOU",  # lounaismurteet, pohjoisryhmä
    "Vihti": "LVÄ",
    "Noormarkku": "LVÄ",
    "Alastaro": "LVÄ",
    "Lieto": "LVÄ",  # lounaiset välimurteet
    "Palkane": "HÄM",
    "Sippola": "HÄM",
    "Ikaalinen": "HÄM",
    "Loppi": "HÄM",
    "Hollola": "HÄM",
    "Askola": "HÄM",
    "Urjala": "HÄM",
    "Artjarvi": "HÄM",
    "Padasjoki": "HÄM",
    "Jaala": "HÄM",
    "Juupajoki": "HÄM",
    "Heinola": "HÄM",
    "Jurva": "POH",
    "Kauhava": "POH",
    "Ilmajoki": "POH",  # Etelä-Pohjanmaan murre
    "Hailuoto": "POH",
    "Paavola": "POH",
    "Kalajoki": "POH",  # Pohjois- and Keski-Pohjanmaan murteet
    "Salla": "POH",
    "Alatornio": "POH",
    "Vittanki": "POH",  # Peräpohja
    "Suomussalmi": "SAV",
    "Lappajarvi": "SAV",
    "Jamsa": "SAV",
    "Lapinlahti": "SAV",
    "Mikkeli": "SAV",
    "Pihtipudas": "SAV",
    "Kiihtelysvaara": "SAV",
    "Liperi": "SAV",
    "Rautalampi": "SAV",
    "Sotkamo": "SAV",
    "Kerimaki": "SAV",
    "Joutsa": "SAV",
    "Pielisjarvi": "SAV",
    "Kitee": "SAV",
    "Saarijarvi": "SAV",
    "Vermlanti": "SAV",
    "Rautu": "KAA",
    "Kurkijoki": "KAA",
    "Lappee": "KAA",  # varsinaiset kaakkoismurteet
    "Moloskovitsa": "KAA",
    "Hietamaki": "KAA",
    "Markkova": "KAA",  # Inkerin suomalaismurteet
}


def lamurre_regions(path: Path | None = None) -> dict[str, str]:
    """LA-murre recording filename -> region, or {} before `prepare lamurre-regions` has run."""
    path = path or DATA / "lamurre_meta" / "regions.tsv"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, region, *_ = line.split("\t")
        if region in NAMES:
            out[name] = region
    return out


def region_of(segment_id: str, lamurre: dict[str, str]) -> str | None:
    """The region of an SKN id (SKN07b_Lappajarvi_0000.wav) or a LA-murre id (hame_tamm1_0000.wav)."""
    head = segment_id.removesuffix(".wav").rsplit("_", 1)[0]
    if head.startswith("SKN"):
        return SKN.get(head.partition("_")[2])
    return lamurre.get(head)


def prompt_text(region: str) -> str:
    return f"Murre: {NAMES[region]}."


def write_lamurre_regions(out: Path | None = None) -> Path:
    """Download LA-murre's public VRT package and write filename, region, group, parish per recording."""
    import io
    import re
    import urllib.request
    import zipfile

    out = out or DATA / "lamurre_meta" / "regions.tsv"
    url = "https://www.kielipankki.fi/download/la-murre/vrt/la-murre-vrt.zip"
    with urllib.request.urlopen(url) as r:
        package = zipfile.ZipFile(io.BytesIO(r.read()))
    rows = []
    for name in package.namelist():
        for line in package.read(name).decode("utf-8").splitlines():
            if line.startswith("<text "):
                a = dict(re.findall(r'(\w+)="([^"]*)"', line))
                rows.append("\t".join(a.get(k, "") for k in ("filename", "dialect_region", "dialect_group", "parish")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{r}\n" for r in sorted(rows)), encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    path = write_lamurre_regions()
    print(f"{path}: {len(lamurre_regions(path))} recordings with a region")
