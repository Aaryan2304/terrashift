"""Render the technical report from Markdown to PDF.

The report is written once, in `docs/report.md`, and rendered from that file.
Nothing is retyped for the PDF, so the two cannot drift.

Table support comes from the mistune table plugin; page styling is CSS for
WeasyPrint. Output is never overwritten.
"""

import argparse
import os
import sys

import mistune
from weasyprint import HTML

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
"""


def render(markdown_path, pdf_path, title):
    with open(markdown_path, encoding="utf-8") as handle:
        text = handle.read()
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(text)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>%s</title><style>%s</style></head><body>%s</body></html>"
        % (title, PAGE_CSS, body))
    HTML(string=html, base_url=os.path.dirname(markdown_path)).write_pdf(pdf_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(ROOT, "docs", "report.md"))
    parser.add_argument("--output", default=os.path.join(ROOT, "docs", "report.pdf"))
    parser.add_argument("--title", default="TerraShift — technical report")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error("input not found: %s" % args.input)
    if os.path.exists(args.output):
        parser.error("refusing to overwrite %s" % args.output)

    render(args.input, args.output, args.title)
    print("wrote", args.output, os.path.getsize(args.output), "bytes")


if __name__ == "__main__":
    sys.exit(main())
