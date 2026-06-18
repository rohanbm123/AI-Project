import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(pipeline, filename="Custom Dataset"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Premium styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1e1b4b"),  # Dark purple
        spaceAfter=5
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#6366f1"),  # Accent purple
        textTransform='uppercase',
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    status = pipeline.compliance_report.get("final_status", "UNKNOWN")
    status_bg = "#dcfce7" if status == "APPROVED" else ("#fef3c7" if status == "CONDITIONALLY APPROVED" else "#fee2e2")
    status_fg = "#15803d" if status == "APPROVED" else ("#b45309" if status == "CONDITIONALLY APPROVED" else "#b91c1c")
    
    status_style = ParagraphStyle(
        'StatusText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor(status_fg),
        alignment=1
    )

    story = []
    
    # Header
    story.append(Paragraph("Anchor AI Compliance Certificate", title_style))
    story.append(Paragraph("Automated Governance Ledger  |  NIST AI RMF 1.0 Alignment", subtitle_style))
    story.append(Spacer(1, 8))
    
    # Executive Summary Table
    score = pipeline.compliance_report.get("score", 0)
    max_score = pipeline.compliance_report.get("max_score", 4)
    eu_risk_tier = pipeline.compliance_report.get("eu_risk_tier", "Low")
    risk = "LOW" if score == max_score else ("MEDIUM" if score >= max_score / 2 else "HIGH")
    
    summary_data = [
        [Paragraph("Executive Governance Summary", bold_body_style), ""],
        [Paragraph("Target Dataset:", body_style), Paragraph(filename, body_style)],
        [Paragraph("Assessment Framework:", body_style), Paragraph("NIST AI Risk Management Framework (AI RMF 1.0)", body_style)],
        [Paragraph("EU AI Act Risk Classification:", body_style), Paragraph(f"{eu_risk_tier}-Risk Sector Tier", body_style)],
        [Paragraph("Model Type (Lead Family):", body_style), Paragraph("Logistic Regression / HistGradientBoosting" if pipeline.task_type == "classification" else "RidgeCV / HistGradientBoosting", body_style)],
        [Paragraph("Target Column (Y):", body_style), Paragraph(pipeline.target_col, body_style)],
        [Paragraph("Demographic Attribute (G):", body_style), Paragraph(pipeline.demographic_col or "None Selected (Audits Skipped)", body_style)],
        [Paragraph("Policy Gate Score:", body_style), Paragraph(f"{score} / {max_score} NIST check rules passed (Overall Risk: {risk})", body_style)],
        [Paragraph("Final Governance Status:", body_style), Paragraph(status, status_style)]
    ]
    
    summary_table = Table(summary_data, colWidths=[150, 350])
    summary_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (1, 8), (1, 8), colors.HexColor(status_bg)),
        ('BOX', (1, 8), (1, 8), 1, colors.HexColor(status_fg)),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 10))
    
    # NIST AI RMF Alignment Checklist
    story.append(Paragraph("NIST AI RMF 1.0 Audit Checklist", h1_style))
    
    nist_headers = [Paragraph("AI RMF Pillar", bold_body_style), Paragraph("Registered Check Rule", bold_body_style), Paragraph("Status", bold_body_style)]
    nist_rows = []
    
    gate_rules = pipeline.compliance_report.get("gate_rules", [])
    for rule in gate_rules:
        r_status = rule.get("status", "N/A")
        s_color = "#15803d" if r_status == "PASS" else ("#b45309" if r_status == "WARNING" else "#b91c1c")
        
        rule_status_style = ParagraphStyle(
            f'RuleStatus_{rule["rule"].replace(" ", "_")}',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor(s_color)
        )
        
        # Format rule name safely to avoid ReportLab superscript parsing errors
        rule_name = rule["rule"].replace("R²", "R<super>2</super>")
        
        nist_rows.append([
            Paragraph(rule.get("nist_pillar", "MEASURE"), body_style),
            Paragraph(f"<b>{rule_name}</b><br/>{rule.get('explanation', '')}", body_style),
            Paragraph(r_status, rule_status_style)
        ])
        
    nist_data = [nist_headers] + nist_rows
    nist_table = Table(nist_data, colWidths=[80, 340, 80])
    nist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(nist_table)
    story.append(Spacer(1, 10))
    
    # Detailed Pillar Verification Sub-tables
    
    # ── MEASURE PILLAR: Performance Metrics ──
    story.append(Paragraph("MEASURE: Model Validation Performance", h1_style))
    perf_headers = [Paragraph("Metric", bold_body_style), Paragraph("Linear Model Value", bold_body_style), Paragraph("Ensemble Model Value", bold_body_style)]
    perf_rows = []
    
    if pipeline.task_type == "classification":
        acc_lin = pipeline.metrics.get("logistic_regression", {}).get("accuracy", 0.0)
        rec_lin = pipeline.metrics.get("logistic_regression", {}).get("recall", 0.0)
        f1_lin = pipeline.metrics.get("logistic_regression", {}).get("f1_score", 0.0)
        
        acc_ens = pipeline.metrics.get("random_forest", {}).get("accuracy", 0.0)
        rec_ens = pipeline.metrics.get("random_forest", {}).get("recall", 0.0)
        f1_ens = pipeline.metrics.get("random_forest", {}).get("f1_score", 0.0)
        
        perf_rows.append([Paragraph("Accuracy", body_style), Paragraph(f"{acc_lin * 100:.1f}%", body_style), Paragraph(f"{acc_ens * 100:.1f}%", body_style)])
        perf_rows.append([Paragraph("Recall", body_style), Paragraph(f"{rec_lin * 100:.1f}%", body_style), Paragraph(f"{rec_ens * 100:.1f}%", body_style)])
        perf_rows.append([Paragraph("F1 Score", body_style), Paragraph(f"{f1_lin * 100:.1f}%", body_style), Paragraph(f"{f1_ens * 100:.1f}%", body_style)])
    else:
        r2_lin = pipeline.metrics.get("linear_regression", {}).get("r2_score", 0.0)
        mae_lin = pipeline.metrics.get("linear_regression", {}).get("mae", 0.0)
        rmse_lin = pipeline.metrics.get("linear_regression", {}).get("rmse", 0.0)
        
        r2_ens = pipeline.metrics.get("random_forest", {}).get("r2_score", 0.0)
        mae_ens = pipeline.metrics.get("random_forest", {}).get("mae", 0.0)
        rmse_ens = pipeline.metrics.get("random_forest", {}).get("rmse", 0.0)
        
        # ReportLab: Use R<super>2</super> instead of superscript ² unicode character
        perf_rows.append([Paragraph("R<super>2</super> Score", body_style), Paragraph(f"{r2_lin:.4f}", body_style), Paragraph(f"{r2_ens:.4f}", body_style)])
        perf_rows.append([Paragraph("MAE", body_style), Paragraph(f"{mae_lin:.4f}", body_style), Paragraph(f"{mae_ens:.4f}", body_style)])
        perf_rows.append([Paragraph("RMSE", body_style), Paragraph(f"{rmse_lin:.4f}", body_style), Paragraph(f"{rmse_ens:.4f}", body_style)])
        
    perf_data = [perf_headers] + perf_rows
    perf_table = Table(perf_data, colWidths=[200, 150, 150])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 8))
    
    # ── MANAGE PILLAR: Fairness & Mitigation ──
    story.append(Paragraph("MANAGE: Fairness Audit & Bias Mitigation", h1_style))
    
    mitigation_status = "Active (Calders-Kamiran Reweighing)" if pipeline.metadata.get("mitigation_applied", False) else "Inactive (Thresholds Satisfied)"
    di_ratio = pipeline.fairness_report.get("disparate_impact_ratio", 1.0)
    gap = pipeline.fairness_report.get("demographic_gap", 0.0)
    
    fair_data = [
        [Paragraph("Audited Attribute", bold_body_style), Paragraph("Selection Rate Gap", bold_body_style), Paragraph("Disparate Impact Ratio", bold_body_style), Paragraph("Mitigation Loop Status", bold_body_style)],
        [Paragraph(pipeline.demographic_col or "None Selected", body_style), Paragraph(f"{gap:.4f}", body_style), Paragraph(f"{di_ratio:.4f}", body_style), Paragraph(mitigation_status, body_style)]
    ]
    fair_table = Table(fair_data, colWidths=[130, 120, 120, 130])
    fair_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(fair_table)
    story.append(Spacer(1, 12))
    
    # Footer Disclaimer
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#94a3b8"),
        alignment=1
    )
    story.append(Paragraph("This certificate is automatically generated by the Anchor AI governance engine based on compiled model training, validation, and fairness audits. Mapped directly to the NIST AI Risk Management Framework (AI RMF 1.0) and EU AI Act risk guidelines.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
