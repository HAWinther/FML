#!/usr/bin/env bash
#=====================================================================================================================
# Download the baccoemu networks and convert them to the files Bacco.h reads (data/).
#
#   ./get_data.sh                 everything: matter emulators (~10 MB of downloads) and the Lagrangian bias
#                                 expansion emulators (~95 MB)
#   ./get_data.sh --matter        only the matter emulators (MatterPowerSpectrum, MatterBispectrum)
#   ./get_data.sh --clean         also delete the downloaded tarballs afterwards
#   ./get_data.sh --python PY     the Python to run export_bacco.py with (default python3; needs numpy and h5py)
#
# The networks of baccoemu 2.3.0 (R. Angulo, M. Zennaro, G. Arico, M. Pellejero Ibanez, P. Burger; MIT licence, see
# LICENSE) are downloaded from https://bacco.dipc.org into weights/ and checked against the SHA-256 sums below. The
# server's certificate chain is incomplete, so curl / wget are run without certificate verification (as baccoemu
# itself does); the checksums make sure the files are the expected ones.
#=====================================================================================================================
set -euo pipefail

URL=${BACCO_URL:-https://bacco.dipc.org} # BACCO_URL: a mirror with the same files
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="$HERE/weights"
DATA="$HERE/data"
PYTHON=python3
MATTER_ONLY=0
CLEAN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --matter) MATTER_ONLY=1 ;;
        --clean) CLEAN=1 ;;
        --python) shift; PYTHON="$1" ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "get_data.sh: unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

# sha256  file  group
FILES="
566446c943207983967a084277bdf930c1121e4691fb6a3beecea49f278fcce9 cold_matter_linear_emu_1.0.2.tar matter
a75e35d534af01c3ea547f63dfe30a659d1476f1b2896db7c42bfb1bfc624690 total_matter_linear_emu_1.0.2.tar matter
de667c59a09ed8b89b0518f1c208eb547bb2ceaf2a7e312eefa89db80020d949 sigma8_emu_2.0.2.tar matter
7efc65a15320abfdf720fe9ad3cb6e1b975fdfd7688618ab05c8239cbeed49a4 smeared_bao_emu_1.0.2.tar matter
2a2e1038ed6ba18d351c1f8e3df000b02a3af5c24a76c141a5a39ffece520597 no_wiggles_emu_1.0.2.tar matter
30568f0f6ec138854d140813b8672660559761a7fedda17ef324df661ced018b nonlinear_emu_1.0.3.tar matter
e6262ad1db34ba18f7da2752901ec095aa9adc668f01ce7806f1b2e95f8d2ec5 nonlinear_emu_2.0.1.tar matter
735238eff376c9bf6670051207c0391ce0410e5169c8468529eb3dd7776f16a8 baryon_emu_1.0.2.tar matter
fc770b24161c68b10e4f53efaa4e28b5bdc42d70c22233487481c7d57aa0b353 bispectrum_baryonic_boost.zip matter
2936eb2243de32bec62a230f33b26f95c910e93968b9698ee43891f3e877757f lpt_emulator_v2.0.2.tar bias
703c8f05d1a4212da861744d9e4534fb6a8d2bf08729cb05e56604b64c598d63 lbias_real_1.0.1.tar bias
7bdb3442c75a44d7be2fecbda10d10d290315a4699fcb955080b28cf137b0359 velocileptor_emulator_v1.0.2.tar bias
f6e41fe4bd989911e14d0cfdc7320e190a12f514fe70974a078b638e27772f0d zspace_smeared_bao_emulator_v1.0.2.tar bias
cc82569b20c98411f2b4759258eb3edd03e927415ecadde777efc69ffb3ef6d1 lbias_emulator_RSD_1.0.2.tar bias
"

sha256() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum > /dev/null; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo "get_data.sh: need sha256sum or shasum" >&2
        exit 1
    fi
}

download() { # url file
    if command -v curl > /dev/null; then
        curl -kfL --retry 3 -o "$2" "$1"
    elif command -v wget > /dev/null; then
        wget --no-check-certificate -O "$2" "$1"
    else
        echo "get_data.sh: need curl or wget" >&2
        exit 1
    fi
}

# The exporter needs numpy and h5py: check before downloading
if ! "$PYTHON" -c "import numpy, h5py" 2> /dev/null; then
    echo "get_data.sh: $PYTHON needs numpy and h5py to convert the networks:" >&2
    echo "    $PYTHON -m pip install --user numpy h5py      (or use --python with another Python)" >&2
    exit 1
fi

mkdir -p "$WEIGHTS" "$DATA"
echo "$FILES" | while read -r sum file group; do
    [ -z "$file" ] && continue
    [ "$MATTER_ONLY" = 1 ] && [ "$group" = bias ] && continue
    out="$WEIGHTS/$file"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$sum" ]; then
        echo "have        $file"
        continue
    fi
    echo "downloading $file"
    download "$URL/$file" "$out.part"
    got="$(sha256 "$out.part")"
    if [ "$got" != "$sum" ]; then
        rm -f "$out.part"
        echo "get_data.sh: wrong checksum for $file (got $got): the file on $URL has changed." >&2
        echo "    baccoemu may have released new networks; FML's Bacco.h is tested with the ones listed here." >&2
        exit 1
    fi
    mv "$out.part" "$out"
done

echo "converting to $DATA"
"$PYTHON" "$HERE/export_bacco.py" "$WEIGHTS" "$DATA"
rm -rf "$WEIGHTS/extracted"
if [ "$CLEAN" = 1 ]; then
    rm -rf "$WEIGHTS"
    echo "deleted the downloaded tarballs"
fi
echo "done: Bacco.h reads $DATA"
