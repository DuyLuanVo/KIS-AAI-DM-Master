"""
Nhung anh thumbnail vao file Excel de xem truc tiep trong Excel.
Su dung sau khi chay compare_captions.py.

Cai dat:
    pip install openpyxl pillow

Cach dung:
    # Mac dinh: xu ly calibration_sample.csv -> calibration_sample.xlsx
    python embed_images_xlsx.py

    # Chi dinh file CSV bat ky (phai co cot 'filepath')
    python embed_images_xlsx.py --input compare_output/manual_review.csv

    # Thay doi kich thuoc anh thu nho (default: rong 120px, cao 80px)
    python embed_images_xlsx.py --width 160 --height 100
"""
import argparse
import sys
import io
from pathlib import Path

try:
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    from PIL import Image as PILImage
except ImportError as e:
    sys.exit(
        f"Thieu thu vien: {e}\n"
        "Vui long cai dat: pip install openpyxl pillow pandas"
    )

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "compare_output"

THUMB_W  = 120
THUMB_H  = 80
COL_W_PX = 130
ROW_H_PT = 65

HEADER_FILL = PatternFill("solid", fgColor="1E3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
THIN = Side(style="thin", color="CCCCCC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def col_px_to_excel(px):
    return px / 7.0


def resize_image(filepath, max_w, max_h):
    try:
        img = PILImage.open(filepath).convert("RGB")
        img.thumbnail((max_w, max_h), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


def build_xlsx(df, out_path, filepath_col="filepath", thumb_w=THUMB_W, thumb_h=THUMB_H):
    wb = Workbook()
    ws = wb.active
    ws.title = "Review"

    cols = list(df.columns)
    if filepath_col in cols:
        insert_pos = cols.index(filepath_col) + 1
    else:
        insert_pos = len(cols)
    img_col_name = "image"
    if img_col_name in cols:
        cols.remove(img_col_name)
    cols.insert(insert_pos, img_col_name)

    for ci, col_name in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=ci, value=col_name)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER

    img_col_idx = cols.index(img_col_name) + 1
    for ci, col_name in enumerate(cols, start=1):
        if ci == img_col_idx:
            ws.column_dimensions[get_column_letter(ci)].width = col_px_to_excel(COL_W_PX)
        elif col_name in ("file",):
            ws.column_dimensions[get_column_letter(ci)].width = 28
        elif col_name in ("similarity", "bin_weight", "ok", "status"):
            ws.column_dimensions[get_column_letter(ci)].width = 12
        elif "caption" in col_name.lower() or "reason" in col_name.lower():
            ws.column_dimensions[get_column_letter(ci)].width = 50
        elif col_name in ("filepath", "path_q", "path_g"):
            ws.column_dimensions[get_column_letter(ci)].width = 40
        else:
            ws.column_dimensions[get_column_letter(ci)].width = 18

    n_imgs = 0
    for ri, (_, row) in enumerate(df.iterrows(), start=2):
        ws.row_dimensions[ri].height = ROW_H_PT
        for ci, col_name in enumerate(cols, start=1):
            if col_name == img_col_name:
                continue
            val = row.get(col_name, "")
            if pd.isna(val):
                val = ""
            cell = ws.cell(row=ri, column=ci, value=str(val) if val != "" else "")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.border = BORDER

        fp = str(row.get(filepath_col, "")).strip()
        if fp and Path(fp).exists():
            img_bytes = resize_image(fp, thumb_w, thumb_h)
            if img_bytes:
                xl_img = XLImage(io.BytesIO(img_bytes))
                col_letter = get_column_letter(img_col_idx)
                xl_img.anchor = f"{col_letter}{ri}"
                ws.add_image(xl_img)
                n_imgs += 1
        ws.cell(row=ri, column=img_col_idx, value="").border = BORDER

    wb.save(out_path)
    return n_imgs


def main():
    ap = argparse.ArgumentParser(description="Nhung anh vao Excel tu compare_captions output")
    ap.add_argument("--input", default=str(OUT_DIR / "manual_review.csv"),
                    help="File CSV dau vao (phai co cot 'filepath')")
    ap.add_argument("--output", default="",
                    help="File xlsx dau ra (mac dinh: thay .csv -> .xlsx)")
    ap.add_argument("--filepath-col", default="filepath",
                    help="Ten cot chua duong dan anh")
    ap.add_argument("--width",  type=int, default=THUMB_W)
    ap.add_argument("--height", type=int, default=THUMB_H)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Khong tim thay file: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".xlsx")
    print(f"Doc: {in_path}")
    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str).fillna("")

    if args.filepath_col not in df.columns:
        sys.exit(f"Khong tim thay cot '{args.filepath_col}'. Cac cot hien co: {list(df.columns)}")

    print(f"Dang xu ly {len(df)} hang, nhung anh {args.width}x{args.height}px...")
    n = build_xlsx(df, out_path, args.filepath_col, args.width, args.height)
    print(f"Da nhung {n}/{len(df)} anh.")
    print(f"Ket qua: {out_path}")


if __name__ == "__main__":
    main()
