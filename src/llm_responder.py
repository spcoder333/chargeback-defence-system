import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

load_dotenv()

# Read API key safely from environment or Streamlit Cloud Secrets
api_key = os.getenv("GEMINI_API_KEY") or (st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None)

client = genai.Client(api_key=api_key)

def assemble_evidence_package(transaction_details, top_reasons, routing_decision):
    prompt = f"""You are assembling a chargeback evidence package for a merchant submitting to a payment gateway like Razorpay.
Organize evidence strictly based on the provided facts and card network standards. Do not invent details.

Transaction details: {transaction_details}
Top contributing risk factors: {top_reasons}
Routing decision: {routing_decision}

Return ONLY valid JSON with this exact schema:
{{
  "transaction_summary": "1-2 sentence factual summary of the transaction",
  "delivery_evidence": "what delivery/fulfillment evidence would support this case",
  "network_reason_analysis": "analysis tailored specifically to card network dispute reason code",
  "customer_history_note": "how customer patterns relate to this dispute based on top reasons",
  "recommendation": "AUTO_SUBMIT",
  "recommended_action_summary": "1 sentence explaining the recommendation"
}}"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return {
            "transaction_summary": "Manual review required — evidence assembly failed",
            "delivery_evidence": "N/A",
            "network_reason_analysis": "N/A",
            "customer_history_note": "N/A",
            "recommendation": "ESCALATE",
            "recommended_action_summary": f"LLM output could not be parsed ({str(e)}); route to human."
        }

def generate_pdf_evidence(package_data, transaction_id):
    """Generates a downloadable binary PDF buffer from the LLM evidence package."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, leading=20)
    bold_style = ParagraphStyle('Bold', parent=styles['Normal'], fontSize=10, leading=14, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14)

    story = [
        Paragraph(f"Chargeback Evidence Package — Case #{transaction_id}", title_style),
        Spacer(1, 15),
        Paragraph(f"<b>Recommendation:</b> {package_data.get('recommendation', 'N/A')}", bold_style),
        Spacer(1, 10),
        Paragraph("<b>Transaction Summary:</b>", bold_style),
        Paragraph(str(package_data.get('transaction_summary', 'N/A')), body_style),
        Spacer(1, 10),
        Paragraph("<b>Delivery & Fulfillment Evidence:</b>", bold_style),
        Paragraph(str(package_data.get('delivery_evidence', 'N/A')), body_style),
        Spacer(1, 10),
        Paragraph("<b>Card Network Reason Analysis:</b>", bold_style),
        Paragraph(str(package_data.get('network_reason_analysis', 'N/A')), body_style),
        Spacer(1, 10),
        Paragraph("<b>Customer Pattern Note:</b>", bold_style),
        Paragraph(str(package_data.get('customer_history_note', 'N/A')), body_style),
        Spacer(1, 10),
        Paragraph("<b>Action Plan:</b>", bold_style),
        Paragraph(str(package_data.get('recommended_action_summary', 'N/A')), body_style),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()