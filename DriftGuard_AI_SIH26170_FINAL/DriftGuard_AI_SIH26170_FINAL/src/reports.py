from __future__ import annotations

from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from config import PARAMETERS
from risk_engine import explain_row, recommendation_for_row


def _base_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="SmallMuted", parent=styles["BodyText"], fontSize=8.5, textColor=colors.HexColor("#667085"), leading=11))
    return styles


def component_report_pdf(row, parameter: str, source_name: str = "") -> bytes:
    cfg = PARAMETERS[parameter]
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = _base_styles()
    story = [
        Paragraph("DriftGuard - Component Reliability Report", styles["CenterTitle"]),
        Spacer(1, 6*mm),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Paragraph(f"Source: {source_name or 'Current analysed batch'}", styles["Normal"]),
        Spacer(1, 4*mm),
    ]

    data = [
        ["Component ID", str(row.get("component_id", ""))],
        ["Lot", str(row.get("lot_id", ""))],
        ["Parameter", f"{cfg.label} ({cfg.unit})"],
        ["0h", f"{float(row.get(f'{parameter}_0h', 0)):.3f}"],
        ["24h", f"{float(row.get(f'{parameter}_24h', 0)):.3f}"],
        ["Predicted 168h", f"{float(row.get('predicted_168h', 0)):.3f} {cfg.unit}"],
        ["Estimated 90% range", f"{float(row.get('prediction_lower_90', 0)):.3f} - {float(row.get('prediction_upper_90', 0)):.3f} {cfg.unit}"],
        ["Risk", f"{float(row.get('risk_score', 0)):.1f}/100 - {row.get('risk_level', '')}"],
        ["Recommendation", recommendation_for_row(row)],
    ]
    table = Table(data, colWidths=[55*mm, 105*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F5F5F7")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D2D2D7")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table, Spacer(1, 6*mm), Paragraph("Why the system reached this decision", styles["Heading2"])]
    for reason in explain_row(row, parameter):
        story.append(Paragraph(f"- {reason}", styles["BodyText"]))

    story += [
        Spacer(1, 5*mm),
        Paragraph("Prototype note", styles["Heading2"]),
        Paragraph(
            "Safety slopes, absolute limits and risk thresholds used here are prototype assumptions for SIH demonstration. "
            "They are not official ISRO or device acceptance limits. Production use requires validated device-specific limits and real laboratory data.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()


def batch_summary_pdf(scored, parameter: str, source_name: str, quality_score: int, summary_text: str) -> bytes:
    """Create a concise, decision-first batch summary report."""
    cfg = PARAMETERS[parameter]
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = _base_styles()

    normal = int((scored["risk_level"] == "Normal").sum())
    warning = int((scored["risk_level"] == "Warning").sum())
    critical = int((scored["risk_level"] == "Critical").sum())
    early = int(scored["early_reject"].sum())

    story = [
        Paragraph("DriftGuard - Engineering Batch Summary", styles["CenterTitle"]),
        Spacer(1, 4*mm),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["SmallMuted"]),
        Paragraph(f"Source: {source_name}", styles["SmallMuted"]),
        Spacer(1, 5*mm),
        Paragraph("Decision summary", styles["Heading2"]),
        Paragraph(summary_text.replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 5*mm),
    ]

    data = [
        ["Parameter", f"{cfg.label} ({cfg.unit})"],
        ["Components analysed", f"{len(scored):,}"],
        ["Normal", str(normal)],
        ["Warning", str(warning)],
        ["Critical", str(critical)],
        ["Early-review flags", str(early)],
        ["Data quality score", f"{quality_score}/100"],
    ]
    table = Table(data, colWidths=[60*mm, 100*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F5F5F7")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D2D2D7")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table, Spacer(1, 6*mm), Paragraph("Priority review queue", styles["Heading2"])]

    top = scored.sort_values("risk_score", ascending=False).head(10)
    queue = [["Component", "Risk", "Predicted 168h", "Recommended action"]]
    for _, row in top.iterrows():
        queue.append([
            str(row["component_id"]),
            f"{float(row['risk_score']):.1f} - {row['risk_level']}",
            f"{float(row['predicted_168h']):.2f} {cfg.unit}",
            recommendation_for_row(row),
        ])
    qtable = Table(queue, colWidths=[34*mm, 34*mm, 42*mm, 55*mm], repeatRows=1)
    qtable.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1D1D1F")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D2D2D7")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(qtable)
    story += [
        Spacer(1, 6*mm),
        Paragraph("Prototype note", styles["Heading2"]),
        Paragraph(
            "This report is generated by a research/hackathon prototype. Thresholds and safety slopes are demonstration assumptions, not official ISRO or device acceptance limits.",
            styles["SmallMuted"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()
