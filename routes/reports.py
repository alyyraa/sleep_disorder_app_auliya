"""Reports dashboard and PDF/Excel export endpoints."""

from io import BytesIO

from flask import Blueprint, render_template, send_file
from flask_login import login_required

from services.excel_report_service import build_excel
from services.pdf_report_service import build_pdf
from services.report_data_service import get_report_data, report_dashboard_cards

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.get("/")
@login_required
def index():
    return render_template("reports/index.html", reports=report_dashboard_cards())


@reports_bp.get("/<string:report_key>/pdf")
@login_required
def export_pdf(report_key):
    report_data = get_report_data(report_key)
    output = BytesIO()
    build_pdf(report_data, output)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f"{report_key}_report.pdf", mimetype="application/pdf")


@reports_bp.get("/<string:report_key>/excel")
@login_required
def export_excel(report_key):
    report_data = get_report_data(report_key)
    output = BytesIO()
    build_excel(report_data, output)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f"{report_key}_report.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
