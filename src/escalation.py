def route_dispute(fraud_confidence, auto_submit_threshold=0.80, auto_reject_threshold=0.20):
    if fraud_confidence >= auto_submit_threshold:
        return {
            "decision": "AUTO_SUBMIT_EVIDENCE",
            "confidence": fraud_confidence,
            "reason": "High-confidence friendly fraud pattern — auto-generate and submit evidence"
        }
    elif fraud_confidence <= auto_reject_threshold:
        return {
            "decision": "LIKELY_LEGITIMATE_DISPUTE",
            "confidence": fraud_confidence,
            "reason": "Low fraud signal — recommend refund, don't fight"
        }
    else:
        return {
            "decision": "ESCALATE_TO_HUMAN",
            "confidence": fraud_confidence,
            "reason": "Ambiguous case — needs manual review before submission"
        }