#!/bin/bash

# make-release.sh
# Empacota e publica o projeto linmod no PyPI via poetry

set -e

echo "🔄 Limpando versões anteriores..."
rm -rf dist/

echo "📦 Construindo com Poetry..."
poetry build

echo "🚀 Publicando no PyPI..."
poetry publish --build
