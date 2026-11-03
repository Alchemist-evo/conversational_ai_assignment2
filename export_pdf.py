"""Export an executed notebook (or an HTML/Markdown report) to PDF via headless Chrome/Edge.

Usage:
  python export_pdf.py Assignment2_PS1_ResearchAssistant.ipynb      -> .html + .pdf next to it
  python export_pdf.py report/Final_Report.html                       -> report/Final_Report.pdf
"""
import shutil
import subprocess
import sys
from pathlib import Path

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "msedge",
]

PRINT_CSS = """
<style>
@page { size: A4; margin: 12mm; }
body { font-size: 11px; }
div.jp-Cell, div.cell { page-break-inside: auto; }
img { max-width: 100% !important; page-break-inside: avoid; }
pre { white-space: pre-wrap !important; word-break: break-word; font-size: 9px; }
table { font-size: 9.5px; }
.jp-OutputArea-output pre { max-height: none !important; }
</style>
"""


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("No Chrome/Edge found for PDF printing")


def to_html(nb_path: Path) -> Path:
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images", str(nb_path)], check=True)
    html = nb_path.with_suffix(".html")
    txt = html.read_text(encoding="utf-8")
    html.write_text(txt.replace("</head>", PRINT_CSS + "</head>", 1), encoding="utf-8")
    return html


def to_pdf(html: Path) -> Path:
    pdf = html.with_suffix(".pdf")
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=20000", f"--print-to-pdf={pdf.resolve()}", html.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pdf


if __name__ == "__main__":
    p = Path(sys.argv[1])
    html = to_html(p) if p.suffix == ".ipynb" else p
    print("PDF:", to_pdf(html))
