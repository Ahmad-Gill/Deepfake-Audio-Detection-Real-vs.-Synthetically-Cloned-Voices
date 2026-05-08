REPORT_MD  := reports/midterm-report-2026-05-08.md
REPORT_TEX := reports/LatexCode.txt
REPORT_PDF := reports/midterm-report-2026-05-08.pdf
LATEX_PDF  := reports/LatexCode.pdf

.PHONY: report report-latex install-deps clean-report

# Generate PDF from Markdown via pandoc + tectonic
report: $(REPORT_PDF)

$(REPORT_PDF): $(REPORT_MD)
	pandoc $< -o $@ \
		--pdf-engine=tectonic \
		--variable geometry:margin=1in \
		--variable fontsize=11pt \
		--variable linestretch=1.25 \
		--syntax-highlighting=tango

# Compile the LaTeX source directly with tectonic
report-latex: $(LATEX_PDF)

$(LATEX_PDF): $(REPORT_TEX)
	tectonic -X compile --outdir reports $<
	mv reports/LatexCode.pdf $(LATEX_PDF) 2>/dev/null || true

# Install tectonic (requires Homebrew)
install-deps:
	brew install tectonic

clean-report:
	rm -f $(REPORT_PDF) $(LATEX_PDF)
