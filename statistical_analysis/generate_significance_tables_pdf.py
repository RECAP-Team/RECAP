"""
Renders all_significance_results.csv into a PDF of per-(language, direction)
tables, in the EXACT visual/textual format of 2025.findings-emnlp.508.pdf's
Tables 11-14 (Appendix A.10.5 "Statistical Significance Testing", page 26):

    Model | DeltachrF++ [95% CI] | p

one table per (language, direction), caption below each table reading
"Table N: DeltachrF++ (95% CI, p-values) vs. <baseline> for <Direction>."
(paper's own caption wording, baseline name swapped to ours). p is shown as
"<0.001" below that threshold (paper's own convention), else 3 decimals;
delta/CI shown with an explicit sign, 2 decimals -- matching the paper's
Tables 11-14 number formatting exactly.

The paper reports NO plots for this section (A.10.5 is tables only) -- so
none are produced here either, to match exactly.

Run (after compute_statistical_significance.py has produced
all_significance_results.csv):
    python generate_significance_tables_pdf.py
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

DIRECTION_LABEL = {
    "hi2tgt": lambda lang: f"Hindi&rarr;{lang}",
    "tgt2hi": lambda lang: f"{lang}&rarr;Hindi",
    "en2tgt": lambda lang: f"English&rarr;{lang}",
    "tgt2en": lambda lang: f"{lang}&rarr;English",
}
DIRECTION_ORDER = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
LANGUAGE_ORDER = ["Bhili", "Mundari", "Gondi"]


def fmt_delta_ci(delta, ci_low, ci_high):
    return f"{delta:+.2f} [{ci_low:.2f}, {ci_high:.2f}]"


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def build_table_html(table_num, lang, direction, baseline, rows):
    direction_label = DIRECTION_LABEL[direction](lang)
    body_rows = "\n".join(
        f"<tr><td>{r['model']}</td><td>{fmt_delta_ci(r['delta_chrF++'], r['ci_low'], r['ci_high'])}</td>"
        f"<td>{fmt_p(r['p_value'])}</td></tr>"
        for r in rows
    )
    return f"""
    <div class="table-block">
      <table>
        <thead>
          <tr><th>Model</th><th>&Delta;chrF++ [95% CI]</th><th>p</th></tr>
        </thead>
        <tbody>
          {body_rows}
        </tbody>
      </table>
      <div class="caption">Table {table_num}: &Delta;chrF++ (95% CI, p-values) vs. {baseline} for {direction_label}.</div>
    </div>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt; color: #000; }}
  h1 {{ font-size: 14pt; text-align: center; margin-bottom: 4pt; }}
  .subtitle {{ text-align: center; font-size: 10pt; color: #444; margin-bottom: 20pt; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 18pt 24pt; }}
  .table-block {{ width: 46%; break-inside: avoid; margin-bottom: 14pt; }}
  table {{ border-collapse: collapse; width: 100%; }}
  thead tr {{ border-top: 1.1pt solid #000; border-bottom: 0.7pt solid #000; }}
  tbody tr:last-child {{ border-bottom: 1.1pt solid #000; }}
  th, td {{ padding: 3pt 6pt; text-align: left; font-size: 9.5pt; }}
  th {{ font-weight: bold; }}
  td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
  .caption {{ font-size: 9pt; margin-top: 4pt; line-height: 1.3; }}
  .lang-header {{ font-size: 12pt; font-weight: bold; margin: 16pt 0 6pt 0; }}
  .lang-header.new-page {{ break-before: page; }}
</style>
</head>
<body>
  <h1>Statistical Significance Testing</h1>
  <div class="subtitle">Paired bootstrap resampling (1,000 iterations), segment-level chrF++ -- replicating 2025.findings-emnlp.508.pdf Appendix A.10.5</div>
  {body}
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "all_significance_results.csv"))
    ap.add_argument("--out", default=str(HERE / "Statistical_Significance_Tables.pdf"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    baseline = df["baseline"].iloc[0]

    body_parts = []
    table_num = 1
    first_lang = True
    for lang in LANGUAGE_ORDER:
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            continue
        header_class = "lang-header" if first_lang else "lang-header new-page"
        first_lang = False
        body_parts.append(f'<div class="{header_class}">{lang}</div>')
        body_parts.append('<div class="grid">')
        for direction in DIRECTION_ORDER:
            rows = lang_df[lang_df["direction"] == direction].to_dict("records")
            if not rows:
                print(f"[skip] {lang}/{direction}: no rows in {args.csv}")
                continue
            body_parts.append(build_table_html(table_num, lang, direction, baseline, rows))
            table_num += 1
        body_parts.append('</div>')

    html = HTML_TEMPLATE.format(body="\n".join(body_parts))
    html_path = Path(args.out).with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    from weasyprint import HTML
    HTML(str(html_path)).write_pdf(args.out)
    print(f"-> {args.out} ({table_num - 1} tables)")


if __name__ == "__main__":
    main()
