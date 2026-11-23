"""Render the technical report from Markdown to PDF.

The report is written once, in Markdown, and rendered from it, so the two cannot drift. ``--input``
is required and points at the source: ``docs/report.md`` here, or an archived copy elsewhere.

Features: mistune tables and strikethrough, a table of contents with real page numbers from a
two-pass build, images scaled to the text column and kept whole across page breaks, and footer page
numbers. Output is never overwritten.

Usage: python scripts/reporting/build_report_pdf.py --input docs/report.md --output docs/report_2026-11-22.pdf
"""

import argparse
import os
import re
import sys

import mistune
from weasyprint import HTML

PAGE_CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 8pt;
    color: #666;
  }
}
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 9.6pt;
  line-height: 1.42;
  color: #111;
}
h1 { font-size: 17pt; margin: 0 0 4mm 0; border-bottom: 0.6pt solid #999; padding-bottom: 2mm; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm 0; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 5mm 0 1.5mm 0; page-break-after: avoid; }
p, li { text-align: justify; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 2mm 0 4mm 0;
  font-size: 8.2pt;
  page-break-inside: avoid;
}
th, td { border: 0.4pt solid #bbb; padding: 1.2mm 1.6mm; text-align: left; vertical-align: top; }
th { background: #efefef; font-weight: bold; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
pre {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 7.8pt;
  background: #f6f6f6;
  border-left: 2pt solid #bbb;
  padding: 2mm 3mm;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
blockquote { margin: 3mm 0 3mm 4mm; padding-left: 4mm; border-left: 2pt solid #ccc; color: #333; }
hr { border: none; border-top: 0.5pt solid #aaa; margin: 5mm 0; }

/* figures: fit the column, never split across a page break */
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 3mm auto 1.5mm auto;
  page-break-inside: avoid;
}

/* table of contents */
nav.toc { page-break-after: always; }
nav.toc h2 { font-size: 13pt; border-bottom: 0.5pt solid #bbb; padding-bottom: 1.5mm; margin-top: 0; }
nav.toc ol { list-style: none; margin: 3mm 0 0 0; padding: 0; font-size: 9.2pt; }
nav.toc li { margin: 0; padding: 0.9mm 0; border-bottom: 0.3pt dotted #ccc; }
nav.toc li.sub { padding-left: 6mm; font-size: 8.8pt; color: #333; }
nav.toc .pg { float: right; font-variant-numeric: tabular-nums; color: #444; }
"""

TITLE = "TerraShift — Man-Made Change Detection Technical Report"


def build_html(markdown_text, toc_numbers, title):
    """Markdown -> HTML with heading anchors and a table of contents.

    ``toc_numbers`` maps an anchor id to a page number string. Pass an empty dict for the first
    pass: the TOC then carries placeholders so pagination for pass two is already correct.
    """
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(markdown_text)

    entries = []

    def add_id(match):
        level = int(match.group(1))
        inner = match.group(2)
        slug = "sec-%d" % len(entries)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        entries.append((level, plain, slug))
        return '<h%d id="%s">%s</h%d>' % (level, slug, inner, level)

    body = re.sub(r"<h([1-3])>(.*?)</h\1>", add_id, body, flags=re.S)

    rows = []
    for level, text, slug in entries:
        if level not in (2, 3):
            continue
        page = toc_numbers.get(slug, "")
        rows.append('<li class="%s">%s<span class="pg">%s</span></li>'
                    % ("sub" if level == 3 else "", text, page))
    toc = ('<nav class="toc"><h2>Contents</h2><ol>%s</ol></nav>' % "".join(rows)) if rows else ""

    # place the TOC after the first top-level heading
    first = body.find("</h1>")
    if first >= 0 and toc:
        body = body[:first + 5] + toc + body[first + 5:]

    return ('<!doctype html><html><head><meta charset="utf-8">'
            "<title>%s</title><style>%s</style></head><body>%s</body></html>"
            % (title, PAGE_CSS, body))


def render(markdown_path, pdf_path, title):
    with open(markdown_path, encoding="utf-8") as handle:
        markdown_text = handle.read()
    base_url = os.path.dirname(os.path.abspath(markdown_path))

    # Pass 1: lay out with placeholder numbers and learn where each heading lands. WeasyPrint
    # exposes page.anchors, so the page numbers are measured, not estimated.
    first_pass = HTML(string=build_html(markdown_text, {}, title), base_url=base_url).render()
    numbers = {}
    for index, page in enumerate(first_pass.pages, start=1):
        for anchor in page.anchors:
            numbers.setdefault(anchor, str(index))

    # Pass 2: rebuild with the measured numbers and write the PDF. This is WeasyPrint's Python
    # Document.write_pdf() method, not the JavaScript document.write().
    document = HTML(string=build_html(markdown_text, numbers, title), base_url=base_url).render()
    document.write_pdf(pdf_path)
    return len(document.pages), len(numbers)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="path to the report Markdown source (required: only the rendered "
                             "PDF is kept in this repository)")
    parser.add_argument("--output", required=True,
                        help="path of the PDF to write; must not already exist")
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error("input not found: %s\n"
                     "The Markdown source is not stored in this repository. Pass the path to the "
                     "archived source, for example:\n"
                     "  --input ../terrashift_local_archive/archived_docs_2026-11-22/docs/report.md"
                     % args.input)
    if os.path.exists(args.output):
        parser.error("refusing to overwrite %s" % args.output)

    pages, anchors = render(args.input, args.output, args.title)
    print("wrote %s (%d bytes, %d pages, %d anchors)"
          % (args.output, os.path.getsize(args.output), pages, anchors))


if __name__ == "__main__":
    sys.exit(main())
