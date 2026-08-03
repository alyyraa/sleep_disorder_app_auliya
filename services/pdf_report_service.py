"""Shared ReportLab document generator using the official report letterhead."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.timezone import format_indonesian_date, jakarta_now

LOGO_PATH = Path(__file__).resolve().parents[1] / "static" / "img" / "logo-mark.png"
BLUE = colors.HexColor("#0B3A75")
LIGHT_BLUE = colors.HexColor("#E8F4FF")

def _page_chrome(canvas, document):
    canvas.saveState()
    width, height = document.pagesize
    logo_x = 18 * mm
    logo_y = height - 48 * mm
    information_x = 52 * mm
    canvas.drawImage(str(LOGO_PATH), logo_x, logo_y, width=28 * mm, height=28 * mm, preserveAspectRatio=True, mask="auto")
    canvas.setFillColor(BLUE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(information_x, height - 21 * mm, "SLEEP STRESS PREDICTOR")
    canvas.setFont("Helvetica", 8.5)
    header_lines = [
        "Sistem Informasi Prediksi Gangguan Tidur dan Tingkat Stres",
        "Berbasis Algoritma XGBoost",
        "",
        "Jl. Pisangan Lama I RT 005 RW 005 No.005, Kel. Pisangan Timur, Kec. Pulogadung",
        "Jakarta Timur 13230 | Telp. 081290143229",
    ]
    y = height - 29 * mm
    for line in header_lines:
        canvas.drawString(information_x, y, line)
        y -= 4.3 * mm
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.5)
    canvas.line(18 * mm, height - 58 * mm, width - 18 * mm, height - 58 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#4E6D8E"))
    canvas.drawString(18 * mm, 14 * mm, f"Halaman {canvas.getPageNumber()}")
    canvas.restoreState()


def build_pdf(report_data, output):
    generated_at = report_data.get("generated_at") or jakarta_now()
    page_size = landscape(A4) if len(report_data["headers"]) > 6 else A4
    document = SimpleDocTemplate(output, pagesize=page_size, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=66 * mm, bottomMargin=25 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], alignment=TA_CENTER, textColor=BLUE, fontSize=15, spaceAfter=8)
    meta_style = ParagraphStyle("ReportMeta", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8, leading=11)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=6.5, leading=8)
    header_style = ParagraphStyle("Header", parent=cell_style, textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold")
    story = [Paragraph(report_data["title"], title_style), Paragraph(f"Tanggal Cetak: {format_indonesian_date(generated_at, include_time=True)}<br/>Total Data: {report_data['total_records']}", meta_style), Spacer(1, 5 * mm)]
    if report_data["summary"]:
        summary_rows = [(label, format_indonesian_date(generated_at, include_time=True) if label == "Tanggal Cetak" else value) for label, value in report_data["summary"]]
        summary_table = Table([[Paragraph(f"<b>{label}</b>", cell_style), Paragraph(str(value), cell_style)] for label, value in summary_rows], colWidths=[55 * mm, 45 * mm])
        summary_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8BDD4")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.extend([summary_table, Spacer(1, 5 * mm)])
    available_width = page_size[0] - document.leftMargin - document.rightMargin
    pdf_headers = report_data["headers"]
    table_rows = [[Paragraph(str(value), header_style) for value in pdf_headers]] + [[Paragraph(str(value), cell_style) for value in row] for row in report_data["rows"]]
    table = Table(table_rows, repeatRows=1, colWidths=[available_width / len(pdf_headers)] * len(pdf_headers))
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8BDD4")), ("BACKGROUND", (0, 1), (-1, -1), colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(table)
    signature = Table([[Paragraph(f"Jakarta, {format_indonesian_date(generated_at)}<br/><br/>Diketahui oleh,<br/><br/><br/>(Auliya Rahmawati)<br/>Administrator", cell_style)]], colWidths=[62 * mm], hAlign="RIGHT")
    signature.spaceBefore = 1 * mm
    signature.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(signature)
    document.build(story, onFirstPage=_page_chrome, onLaterPages=_page_chrome)
