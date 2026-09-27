#!/bin/bash
# Compile and push PDFs for anchoring-llms-empirical-evidence

cd ~/Documents/anchoring-llms-empirical-evidence

echo "Compiling manuscript..."
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex
bibtex manuscript_vih_format_full
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex

if [ -f "manuscript_vih_format_full.pdf" ]; then
    echo "✓ Compilation successful"
    cp manuscript_vih_format_full.pdf manuscript_full.pdf
    
    echo "Pushing to GitHub..."
    git add manuscript_vih_format_full.pdf manuscript_full.pdf
    git commit -m "Update PDFs: add attribution honesty, strengthen OOD limitations, compress to 3,545 words"
    git push
    echo "✓ Done"
else
    echo "✗ Compilation failed"
fi
