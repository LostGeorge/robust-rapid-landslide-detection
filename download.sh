# This script downloads and unzips the datacubes into data/

if [ ! -d "data" ]; then
  mkdir data
fi
cd data

echo "Downloading Kaikoura data..."
wget -O "kaikoura_newzealand.zip" "https://zenodo.org/record/7248056/files/kaikoura_newzealand.zip?download=1"
unzip kaikoura_newzealand.zip
rm kaikoura_newzealand.zip

echo "Downloading Puerto Rico data..."
wget -O "puerto_rico.zip" "https://zenodo.org/record/7248056/files/puerto_rico.zip?download=1"
unzip puerto_rico.zip
rm puerto_rico.zip

echo "Downloading Hokkaido data..."
wget -O "hokkaido_japan.zip" "https://zenodo.org/record/7248056/files/hokkaido_japan.zip?download=1"
unzip hokkaido_japan.zip
rm hokkaido_japan.zip

echo "Downloading Talakmau data..."
wget -O "talakmau_indonesia.zip" "https://zenodo.org/record/7248056/files/talakmau_indonesia.zip?download=1"
unzip talakmau_indonesia.zip
rm talakmau_indonesia.zip
