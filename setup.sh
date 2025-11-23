#!/usr/bin/env bash
set -euo pipefail

echo "[SETUP] Updating apt packages..."
sudo apt-get update -y

echo "[SETUP] Installing build dependencies..."
sudo apt-get install -y cmake build-essential

echo "[SETUP] Installing Python dependencies..."
pip3 install --user -r requirements.txt

echo "[SETUP] Installing GDAL (Geocarpentry script)..."
curl -sL "https://url.geocarpentry.org/gdal-ubuntu" | bash

echo "[SETUP] Setup complete!"
