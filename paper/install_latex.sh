#!/usr/bin/env bash
mkdir -p /tmp/apt-archives /tmp/apt-cache
apt-get install \
    -o Dir::Cache=/tmp/apt-cache \
    -o Dir::Cache::Archives=/tmp/apt-archives \
    -y --fix-missing latexmk texlive-bibtex-extra
