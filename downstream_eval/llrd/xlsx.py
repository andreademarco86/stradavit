from __future__ import annotations

import os
import zipfile
from xml.sax.saxutils import escape

from downstream_eval.llrd.settings import _cfg_label
from downstream_eval.llrd.run_types import EvalMode
from downstream_eval.llrd.runtime import guarded_print


def _xlsx_col_name(idx: int) -> str:
    name = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_sheet_xml(rows: list[list[str]], widths: list[float]) -> str:
    col_xml = []
    for idx, width in enumerate(widths):
        excel_idx = idx + 1
        col_xml.append(
            f'<col min="{excel_idx}" max="{excel_idx}" width="{width:.1f}" customWidth="1"/>'
        )

    row_xml = []
    for r_idx, row in enumerate(rows, 1):
        cells = []
        for c_idx, value in enumerate(row):
            ref = f"{_xlsx_col_name(c_idx)}{r_idx}"
            text = escape(str(value))
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'
            )
        row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<cols>{"".join(col_xml)}</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def _write_simple_xlsx(path: str, sheets: list[tuple[str, list[list[str]], list[float]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (name, _, _) in enumerate(sheets, 1):
        sheet_name = "".join("_" if ch in r'[]:*?/\\' else ch for ch in name)[:31] or f"Sheet{idx}"
        safe_name = escape(sheet_name, {'"': "&quot;"})
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles_xml)
        for idx, (_, rows, widths) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(rows, widths))


def save_rank_table_xlsx(
    sections: list[dict],
    eval_mode_sel: EvalMode,
    *,
    output_dir: str,
    lr_configs: list[dict],
) -> None:
    headers = ["Rank", "Alias", "Macro F1", "Macro Std", "Weighted F1", "Weighted Std"]
    sheets = []
    for section in sections:
        label = str(section["label"])
        sorted_results = sorted(section["results"], key=lambda x: x[1], reverse=True)
        rows = [headers, [label] + [""] * (len(headers) - 1)]
        for rank, rec in enumerate(sorted_results, 1):
            alias = rec[0]
            if isinstance(alias, str):
                alias = alias.split(" cfg=")[0]
                if alias.endswith((" [FT]", " [LP]")):
                    alias = alias[:-5]
            rows.append([
                str(rank),
                str(alias),
                f"{float(rec[1]):.3f}",
                f"{float(rec[2]):.3f}",
                f"{float(rec[3]):.3f}",
                f"{float(rec[4]):.3f}",
            ])

        alias_width = min(80.0, max(18.0, max(len(row[1]) for row in rows) + 2.0))
        sheets.append((label, rows, [8.0, alias_width, 12.0, 12.0, 13.0, 13.0]))

    if not sheets:
        guarded_print("[XLSX] No ranking tables to save.")
        return

    mode_tag = "FT" if eval_mode_sel == EvalMode.FINETUNE else "LP"
    cfg_label = _cfg_label(lr_configs[0]) if lr_configs else "cfg"
    cfg_safe = cfg_label.replace("/", "-").replace(" ", "")
    out_table_xlsx = os.path.join(output_dir, f"rank_table_{mode_tag}_{cfg_safe}.xlsx")
    _write_simple_xlsx(out_table_xlsx, sheets)
    guarded_print(f"Saved ranking table workbook to {out_table_xlsx}")
