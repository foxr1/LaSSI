#!/bin/bash
# Installing all Linux dependencies
sudo apt-get update && apt-get install build-essential python3-dev python3-pip lbzip2 python3.13-venv -y
mkdir -p data

# Creating the Python environment
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
fi

# ConceptNet with 2019 (stale) data
if [ ! -f data/ ]; then
   wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
   mv conceptnet-assertions-5.7.0.csv.gz data/conceptnet_old.csv.gz
fi

# Extracting from Wiktionary
if [ ! -f data/wiktionary.json ]; then
	# Downloading Wiktionary
	if [ ! -f data/enwiktionary-20250220-pages-articles.xml.bz2 ]; then
    		wget https://dumps.wikimedia.org/enwiktionary/20250220/enwiktionary-20250220-pages-articles.xml.bz2
    		mv enwiktionary-20250220-pages-articles.xml.bz2 data/
	fi

    	# Setting up wiktextract
	pushd submodules/wiktextract
	python -m pip install -U pip
	python -m pip install -e .
	popd
	
	# Extraction
	wiktwords --all --language-code en --out data/wiktionary.json --edition en data/enwiktionary-20250220-pages-articles.xml.bz2
fi





