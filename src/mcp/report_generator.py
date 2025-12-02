"""
Network Security Report Generator
Creates professional PDF reports with charts and recommendations
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime
from pathlib import Path
import io
from typing import Dict, List


class NetworkSecurityReportGenerator:
    """Generate comprehensive security reports"""

    def __init__(self, output_dir: str = "/Users/nidhirajani/Desktop/DIC Phase 3/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12
        ))

        self.styles.add(ParagraphStyle(
            name='Alert',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            leftIndent=20
        ))

    def create_pie_chart(self, data: Dict[str, int], title: str) -> str:
        """Create pie chart and return image path"""
        fig, ax = plt.subplots(figsize=(8, 6))

        labels = list(data.keys())
        sizes = list(data.values())
        colors_list = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce', '#85c1e2', '#f8b739', '#52b788']

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list, startangle=90)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Save to bytes
        img_path = self.output_dir / f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(img_path, bbox_inches='tight', dpi=150)
        plt.close()

        return str(img_path)

    def create_bar_chart(self, data: Dict[str, int], title: str, xlabel: str) -> str:
        """Create bar chart and return image path"""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = list(data.keys())
        values = list(data.values())

        bars = ax.bar(labels, values, color='#3498db')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        img_path = self.output_dir / f"barchart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(img_path, bbox_inches='tight', dpi=150)
        plt.close()

        return str(img_path)

    def generate_report(self,
                        network_stats: Dict,
                        malware_distribution: Dict,
                        top_threats: List[Dict],
                        cve_mappings: List[Dict],
                        recommendations: List[str]) -> str:
        """
        Generate comprehensive PDF report

        Args:
            network_stats: {'total': int, 'malicious': int, 'benign': int, 'threat_level': str}
            malware_distribution: {'PortScan': 1000, 'Okiru': 500, ...}
            top_threats: [{'family': str, 'count': int, 'percentage': float}, ...]
            cve_mappings: [{'cve_id': str, 'cvss': float, 'description': str}, ...]
            recommendations: ['Action 1', 'Action 2', ...]
        """

        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"network_security_report_{timestamp}.pdf"

        # Create PDF
        doc = SimpleDocTemplate(str(filename), pagesize=letter,
                                topMargin=0.75 * inch, bottomMargin=0.75 * inch)

        story = []

        # Title Page
        story.append(Paragraph("IoT Network Security Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                               self.styles['Normal']))
        story.append(Spacer(1, 0.5 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        threat_level = network_stats.get('threat_level', 'UNKNOWN')
        threat_color = 'red' if threat_level == 'HIGH' else 'orange' if threat_level == 'MEDIUM' else 'green'

        summary_data = [
            ['Total Traffic Analyzed', f"{network_stats['total']:,} connections"],
            ['Threat Level', threat_level],
            ['Malicious Traffic',
             f"{network_stats['malicious']:,} ({network_stats['malicious'] / network_stats['total'] * 100:.1f}%)"],
            ['Benign Traffic',
             f"{network_stats['benign']:,} ({network_stats['benign'] / network_stats['total'] * 100:.1f}%)"]
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

        # Threat Analysis
        story.append(PageBreak())
        story.append(Paragraph("Threat Analysis", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2 * inch))

        # Malware distribution pie chart
        if malware_distribution:
            chart_path = self.create_pie_chart(malware_distribution, "Malware Family Distribution")
            img = Image(chart_path, width=5 * inch, height=3.75 * inch)
            story.append(img)
            story.append(Spacer(1, 0.3 * inch))

        # Top Threats Table
        story.append(Paragraph("Top Threats Detected", self.styles['Heading3']))
        story.append(Spacer(1, 0.1 * inch))

        threat_data = [['Rank', 'Malware Family', 'Attack Count', 'Percentage']]
        for i, threat in enumerate(top_threats[:5], 1):
            threat_data.append([
                str(i),
                threat['family'],
                f"{threat['count']:,}",
                f"{threat['percentage']:.1f}%"
            ])

        threat_table = Table(threat_data, colWidths=[0.75 * inch, 2 * inch, 1.5 * inch, 1.5 * inch])
        threat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        story.append(threat_table)
        story.append(Spacer(1, 0.3 * inch))

        # Vulnerability Mapping
        story.append(PageBreak())
        story.append(Paragraph("CVE Vulnerability Mapping", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2 * inch))

        if cve_mappings:
            for cve in cve_mappings[:5]:
                story.append(Paragraph(f"<b>{cve['cve_id']}</b> (CVSS: {cve['cvss']})",
                                       self.styles['Heading4']))
                story.append(Paragraph(cve['description'][:300] + "...",
                                       self.styles['Normal']))
                story.append(Spacer(1, 0.15 * inch))
        else:
            story.append(Paragraph("No specific CVE mappings found for detected threats.",
                                   self.styles['Normal']))

        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Remediation Recommendations", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2 * inch))

        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        # Footer
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph(
            "<i>This report was generated automatically by the IoT Malware Detection System using machine learning analysis of network traffic patterns.</i>",
            self.styles['Normal']))
        story.append(Paragraph(f"<i>Model Accuracy: 99.56% | AUC-ROC: 99.94%</i>",
                               self.styles['Normal']))

        # Build PDF
        doc.build(story)

        print(f"✅ Report generated: {filename}")
        return str(filename)

    def generate_markdown_report(self,
                                 network_stats: Dict,
                                 malware_distribution: Dict,
                                 top_threats: List[Dict],
                                 cve_mappings: List[Dict],
                                 recommendations: List[str]) -> str:
        """Generate Markdown version of report"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"network_security_report_{timestamp}.md"

        md_content = f"""# IoT Network Security Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Traffic Analyzed | {network_stats['total']:,} connections |
| Threat Level | **{network_stats['threat_level']}** |
| Malicious Traffic | {network_stats['malicious']:,} ({network_stats['malicious'] / network_stats['total'] * 100:.1f}%) |
| Benign Traffic | {network_stats['benign']:,} ({network_stats['benign'] / network_stats['total'] * 100:.1f}%) |

---

## Top Threats Detected

"""
        for i, threat in enumerate(top_threats[:5], 1):
            md_content += f"{i}. **{threat['family']}**: {threat['count']:,} attacks ({threat['percentage']:.1f}%)\n"

        md_content += "\n---\n\n## CVE Vulnerability Mapping\n\n"

        for cve in cve_mappings[:5]:
            md_content += f"### {cve['cve_id']} (CVSS: {cve['cvss']})\n\n"
            md_content += f"{cve['description'][:300]}...\n\n"

        md_content += "---\n\n## Remediation Recommendations\n\n"

        for i, rec in enumerate(recommendations, 1):
            md_content += f"{i}. {rec}\n"

        md_content += "\n---\n\n*This report was generated automatically by the IoT Malware Detection System.*\n"
        md_content += "*Model Performance: Accuracy 99.56% | AUC-ROC 99.94%*\n"

        with open(filename, 'w') as f:
            f.write(md_content)

        print(f"✅ Markdown report generated: {filename}")
        return str(filename)