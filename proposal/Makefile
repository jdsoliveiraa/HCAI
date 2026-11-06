# HCAI project proposal — local LaTeX build
MAIN    := ijcai27
PDF     := $(MAIN).pdf
PREVIEW := preview
PORT    := 8765

.PHONY: pdf preview text serve watch clean

## Build the PDF with Tectonic (runs bibtex automatically)
pdf: $(PDF)

$(PDF): $(MAIN).tex $(wildcard *.bib *.sty *.bst) $(wildcard sections/*.tex figures/*)
	tectonic --keep-logs --keep-intermediates $(MAIN).tex

## Render every page to PNG under preview/ (for checking layout without a display)
preview: $(PDF)
	@mkdir -p $(PREVIEW) && rm -f $(PREVIEW)/page-*.png
	pdftoppm -r 80 -png $(PDF) $(PREVIEW)/page
	@ls $(PREVIEW)

## Dump text with layout (fast sanity check in the terminal)
text: $(PDF)
	pdftotext -layout $(PDF) -

## Serve the folder; on your laptop run: ssh -L $(PORT):localhost:$(PORT) <devbox>
## then open http://localhost:$(PORT)/$(PDF)
serve:
	@echo "Open http://localhost:$(PORT)/$(PDF) after: ssh -L $(PORT):localhost:$(PORT) <devbox>"
	python3 -m http.server $(PORT) --bind 127.0.0.1

## Rebuild whenever a source file changes (needs inotifywait)
watch:
	while inotifywait -q -e close_write $(MAIN).tex *.bib *.sty; do $(MAKE) pdf; done

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).xdv
	rm -rf $(PREVIEW)
