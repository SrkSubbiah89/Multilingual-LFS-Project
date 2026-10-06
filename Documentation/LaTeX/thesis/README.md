# Reviewed thesis sources

`report.tex` is the main document. Compile active filenames without the
`_ORIGINAL_UNVERIFIED` suffix. The originals and historical audit archives
are for comparison and are not submission versions.

The October 2026 revision improves traceability, academic claims,
methodology, statistical reporting, and layout. It does not certify an
institutional format or establish empirical work that has not been done.
See `THESIS_REVIEW_2026-10-03.md` for corrections and remaining research gaps.

The October 3 review copy described the application as of that date. The
thesis now also reports the fragment-to-parent retrieval configuration
adopted on 6 October (Section 6.1.4 and Appendix run P). Subsequent
application fixes and test evidence are recorded in
[CODE_FIXES_2026-10-05.md](../../CODE_FIXES_2026-10-05.md); the retrieval
change and its accuracy reconciliation are in
[RAG_ACCURACY_AUDIT_2026-10-06.md](../../RAG_ACCURACY_AUDIT_2026-10-06.md).

## Build

Use a current XeLaTeX or LuaLaTeX installation with `latexmk`, `fontspec`,
`polyglossia`, TikZ, and the TeX Gyre Termes/Heros/Cursor fonts. The body is
English and the source is UTF-8. pdfLaTeX is not supported by this preamble.

From this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build report.tex
```

The reviewed build used Tectonic 0.17.0, which runs XeTeX and BibTeX and
reruns until references settle:

```powershell
tectonic --keep-logs --keep-intermediates --outdir build report.tex
```

The workspace-local compiler is at
`../../../Software/thesis_review/tectonic.exe` relative to this folder.
Its first build downloads a LaTeX bundle. Build intermediates belong in
`build/`; `thesis_reviewed.pdf` is the review copy.

## Verify

From the project root, using Python 3.11 or later:

```powershell
python Documentation/LaTeX/thesis/verify_thesis.py
python Documentation/LaTeX/thesis/verify_results.py --output Documentation/LaTeX/thesis/results_verification.json
```

`verify_thesis.py` checks active inputs, labels, citations, environments,
graphic files, and the retained compilation log. `verify_results.py`
recalculates reported results from preserved evaluation artifacts using
the Python standard library. These checks do not validate expert labels,
human outcomes, or international-standard instrument compliance.

The results figure can be regenerated with Python and Matplotlib:

```powershell
python Documentation/LaTeX/thesis/generate_figures.py
```

## Source and evidence policy

- Keep measured predictions and original manifests unchanged. Corrections
  to interpretation are documented separately; new measurements need a
  new run identifier.
- Restrict claims to their configuration, dataset, language, and unit of
  analysis. See Appendix F for recorded run settings and provenance limits.
- The best offline ISCO configuration differs from the configuration the
  application actually serves. The offline best (40.95%) has an unresolved
  executed encoder; the served fragment-to-parent configuration is measured
  separately at 38.83% and is the number to quote for the running system.
  Synthetic interviews do not establish human usability or burden.
- The signed declaration/certificate, final institutional formatting,
  ethics review, and any required disclosure of editing assistance remain
  the author's and institution's responsibilities.

Application defects and the tests run for the separate code review are
recorded in `Documentation/CODE_REVIEW_2026-10-03.md` at the project root.
