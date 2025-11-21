#!/usr/bin/env bash
set -e

SRC="$1"

if [ -z "$SRC" ]; then
  echo "Brug: $0 /sti/til/gammel/data"
  echo "Fx:  $0 /Users/dennishs/Documents/padeledge/data"
  exit 1
fi

if [ ! -d "$SRC/samples" ]; then
  echo "Forventede mappen '$SRC/samples' men fandt den ikke."
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PROJECT_DIR/data"
rsync -av "$SRC/" "$PROJECT_DIR/data/"

echo
echo "✅ Træningsdata kopieret til:"
echo "   $PROJECT_DIR/data"
echo "Docker vil nu se filerne via volume-mount ./data:/app/data"
