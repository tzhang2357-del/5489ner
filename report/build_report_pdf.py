"""
report/build_report_pdf.py
将 report_draft.md 转为自包含 HTML，并用 Edge headless 打印为 PDF。

用法（从仓库根目录或任意目录）:
    python report/build_report_pdf.py

输出:
    report/report.html
    report/NER_Report.pdf
"""

import subprocess
import sys
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).parent
MD = REPORT_DIR / "report_draft.md"
HTML = REPORT_DIR / "report.html"
PDF = REPORT_DIR / "NER_Report.pdf"

CSS = """
body { font-family: 'Times New Roman', 'Microsoft YaHei', serif;
       max-width: 860px; margin: 2em auto; line-height: 1.55;
       color: #222; font-size: 11.5pt; }
h1 { font-size: 20pt; }
h2 { font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; }
h3 { font-size: 12.5pt; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 3px 8px; }
th { background: #f0f0f0; }
img { max-width: 100%; }
code { background: #f5f5f5; padding: 1px 4px;
       font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f5f5f5; padding: 8px; overflow-x: auto; }
pre code { padding: 0; }
blockquote { color: #555; border-left: 3px solid #ccc;
             margin-left: 0; padding-left: 1em; }
"""


def main():
    text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    HTML.write_text(html, encoding="utf-8")
    print("HTML written:", HTML)

    edge_candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    edge = next((p for p in edge_candidates if p.exists()), None)
    if edge is None:
        print("Edge not found. Open report.html manually and print to PDF.")
        return 1

    cmd = [str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={PDF}", HTML.as_uri()]
    subprocess.run(cmd, check=False)
    if PDF.exists():
        print("PDF written:", PDF)
        return 0
    print("PDF generation failed; open report.html manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())