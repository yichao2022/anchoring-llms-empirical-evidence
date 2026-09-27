#!/bin/bash
# Compile manuscript for Value in Health submission

cd /Users/cary/Documents/anchoring-llms-empirical-evidence

# Compile manuscript
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex
bibtex manuscript_vih_format_full
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex
pdflatex -interaction=nonstopmode manuscript_vih_format_full.tex

# Check if PDF was created successfully
if [ -f "manuscript_vih_format_full.pdf" ]; then
    echo "✓ Compilation successful"
    ls -lh manuscript_vih_format_full.pdf
    
    # Copy to manuscript_full.pdf for GitHub
    cp manuscript_vih_format_full.pdf manuscript_full.pdf
    
    # Stage and commit
    git add manuscript_vih_format_full.pdf manuscript_full.pdf
    git commit -m "Update PDFs: add attribution honesty, strengthen OOD limitations, compress to 3,545 words"
    git push
    
    echo "✓ PDFs pushed to GitHub"
else
    echo "✗ Compilation failed"
fi