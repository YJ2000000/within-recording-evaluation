#!/bin/bash
# CogLoad + Snake archive (RAR v4). macOS tar (bsdtar/libarchive) can extract it.
set -e
D=${DATA_ROOT:-/path/to/project/data/raw}/CogLoad
mkdir -p "$D/ext" && cd "$D"
[ -f CogDatasets.rar ] || curl -sSL -o CogDatasets.rar https://martingjoreski.github.io/files/CogDatasets.rar
tar -xf CogDatasets.rar -C ext && echo "CogLoad extracted: $(find ext -type f | wc -l) files"
