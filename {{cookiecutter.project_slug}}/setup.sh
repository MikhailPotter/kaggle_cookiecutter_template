#!/bin/bash

# Download competition data
echo "Downloading competition data..."
kaggle competitions download -c {{ cookiecutter.competition_url | replace('https://www.kaggle.com/c/', '') }}

# Unzip the downloaded files
echo "Unzipping files..."
unzip '*.zip' -d data/raw
rm *.zip

echo "Setup complete!"
