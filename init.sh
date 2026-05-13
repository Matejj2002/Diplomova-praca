#!/bin/bash
clear

source .venv/Scripts/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtuálne prostredie sa nepodarilo aktivovať."
else
    echo "✅ Virtuálne prostredie je aktivované."
    echo "Cesta: $VIRTUAL_ENV"
fi