from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import datetime

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
    def generate_report(self, results):
        # Create PDF document
        doc = SimpleDocTemplate(
            f"game_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            pagesize=letter
        )
        
        # Create content
        content = []
        
        # Add title
        title = Paragraph("Game Report", self.styles['Title'])
        content.append(title)
        
        # Add results table
        data = [
            ["Player Score", str(results['player_score'])],
            ["Computer Score", str(results['computer_score'])],
            ["Rounds Played", str(results['rounds_played'])]
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        
        # Build PDF
        doc.build(content) 