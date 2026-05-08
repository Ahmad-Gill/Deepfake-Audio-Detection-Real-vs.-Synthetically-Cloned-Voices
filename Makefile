REPORT_MD  := reports/midterm-report-2026-05-08.md
REPORT_PDF := reports/midterm-report-2026-05-08.pdf

.PHONY: report clean-report

report: $(REPORT_PDF)

$(REPORT_PDF): $(REPORT_MD)
	pandoc $< -o $@ \
		--pdf-engine=xelatex \
		--variable geometry:margin=1in \
		--variable fontsize=11pt \
		--variable linestretch=1.25 \
		--highlight-style=tango

clean-report:
	rm -f $(REPORT_PDF)
