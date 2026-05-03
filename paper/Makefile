# Compile the paper locally.
#
# Requirements (Ubuntu/Debian):
#   sudo apt-get install texlive-latex-recommended texlive-latex-extra \
#                        texlive-fonts-recommended texlive-bibtex-extra \
#                        latexmk
#
# On macOS, install MacTeX: https://tug.org/mactex/
# On Windows, install MiKTeX: https://miktex.org/

MAIN = main

.PHONY: all clean

all:
	tectonic $(MAIN).tex

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).run.xml
