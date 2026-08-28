import os
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

class EvidencePackage(BaseModel):
    transaction_summary: str = Field(description="1-2 sentence factual summary of the transaction")
    delivery_evidence: str = Field(description="what delivery/fulfillment evidence would support this case")
    network_reason_analysis: str = Field(description="analysis tailored specifically to the card network dispute reason code")
    customer_history_note: str = Field(description="how customer patterns relate to this dispute, based on top reasons")
    recommendation: str = Field(description="AUTO_SUBMIT, ESCALATE, or DONT_FIGHT")
    recommended_action_summary: str = Field(description="1 sentence explaining the recommendation")

def assemble_evidence_package(transaction_details, top_reasons, routing_decision):
    prompt = f"""You are assembling a chargeback evidence package for a merchant submitting to a payment gateway like Razorpay.
Organize evidence strictly based on the provided facts and card network standards.

Transaction details: {transaction_details}
Top contributing risk factors: {top_reasons}
Routing decision: {routing_decision}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EvidencePackage,
            },
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
            "recommended_action_summary": "LLM output could not be parsed; route to human."
        }