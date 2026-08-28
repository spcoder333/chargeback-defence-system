import os
import sys
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from escalation import route_dispute
from llm_responder import assemble_evidence_package

st.set_page_config(page_title="Razorpay AI Chargeback Defense System", layout="wide")

model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
metrics = joblib.load(os.path.join(BASE_DIR, 'models', 'metrics.pkl'))
feature_importance = pd.read_csv(os.path.join(BASE_DIR, 'models', 'feature_importance.csv'), index_col=0)
test_sample = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test_sample.csv'))

st.title("Razorpay AI Chargeback Defense Engine")
st.caption("Automated threat scoring & evidence generation powered by XGBoost & Google Gemini")

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard Overview", "Analyze Dispute", "Webhook Simulator", "Model Metrics"])

with tab1:
    st.subheader("Automated Triage Overview")
    test_encoded = pd.get_dummies(test_sample.drop(columns=['transaction_id', 'is_chargeback_fraud']), columns=['payment_mode', 'merchant_category', 'dispute_reason_code'], drop_first=True)
    test_encoded = test_encoded.reindex(columns=feature_names, fill_value=0)
    
    probas = model.predict_proba(test_encoded)[:, 1]
    decisions = [route_dispute(p)['decision'] for p in probas]
    decision_counts = pd.Series(decisions).value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disputes", len(test_sample))
    col2.metric("Auto-Submitted", decision_counts.get("AUTO_SUBMIT_EVIDENCE", 0))
    col3.metric("Escalated to Human", decision_counts.get("ESCALATE_TO_HUMAN", 0))
    col4.metric("Likely Legitimate", decision_counts.get("LIKELY_LEGITIMATE_DISPUTE", 0))

    st.bar_chart(decision_counts)

with tab2:
    st.subheader("Dispute Resolution Desk")
    selected_tx_id = st.selectbox("Select Transaction ID", test_sample['transaction_id'])
    
    row = test_sample[test_sample['transaction_id'] == selected_tx_id].iloc[0]
    st.json({
        "Transaction ID": row['transaction_id'],
        "Amount (INR)": f"₹{row['amount']}",
        "Payment Mode": row['payment_mode'],
        "Merchant Category": row['merchant_category'],
        "Network Reason Code": row['dispute_reason_code'],
        "IP / Shipping Match": "Yes" if row['ip_shipping_match'] == 1 else "No",
        "VPN Used": "Yes" if row['is_vpn_used'] == 1 else "No",
        "Past Disputes": row['past_dispute_count']
    })

    if st.button("Run Chargeback Defense Analysis"):
        tx_df = pd.DataFrame([row.drop(['transaction_id', 'is_chargeback_fraud'])])
        tx_encoded = pd.get_dummies(tx_df, columns=['payment_mode', 'merchant_category', 'dispute_reason_code'], drop_first=True)
        tx_encoded = tx_encoded.reindex(columns=feature_names, fill_value=0)

        fraud_confidence = float(model.predict_proba(tx_encoded)[0][1])
        routing = route_dispute(fraud_confidence)

        st.metric("Fraud Confidence Score", f"{fraud_confidence:.1%}")
        st.info(f"**Action:** `{routing['decision']}` — {routing['reason']}")

        st.write("**Top Contributing Risk Factors:**")
        st.bar_chart(feature_importance.head(5))

        if routing['decision'] != "LIKELY_LEGITIMATE_DISPUTE":
            with st.spinner("Generating evidence package via Gemini..."):
                package = assemble_evidence_package(
                    transaction_details=row.to_dict(),
                    top_reasons=list(feature_importance.head(5).index),
                    routing_decision=routing['decision']
                )
            st.subheader("Generated Evidence Package")
            st.write("**Summary:**", package['transaction_summary'])
            st.write("**Delivery Evidence:**", package['delivery_evidence'])
            st.write("**Network Reason Analysis:**", package['network_reason_analysis'])
            st.write("**Customer Pattern:**", package['customer_history_note'])
            st.write("**Action Plan:**", package['recommended_action_summary'])

with tab3:
    st.subheader("Live Webhook Event Simulator")
    st.write("Simulate a real-time `dispute.created` HTTP webhook event from Razorpay.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        sim_amount = st.number_input("Amount (INR)", value=4500.0)
        sim_mode = st.selectbox("Payment Mode", ['UPI', 'CREDIT_CARD', 'DEBIT_CARD', 'NETBANKING', 'WALLET'])
        sim_category = st.selectbox("Category", ['ELECTRONICS', 'GAMING', 'RETAIL', 'TRAVEL', 'DIGITAL_SERVICES'])
        sim_reason = st.selectbox("Dispute Reason", ['10.4_FRAUDULENT_TRANSACTION', '13.1_MERCHANDISE_NOT_RECEIVED', '13.3_NOT_AS_DESCRIBED'])
    with col_b:
        sim_match = st.checkbox("IP matches Shipping Address", value=False)
        sim_vpn = st.checkbox("VPN Detected", value=True)
        sim_failed = st.slider("Failed Attempts (24h)", 0, 10, 3)
        sim_disputes = st.slider("Past Dispute Count", 0, 5, 1)

    if st.button("Trigger Webhook Payload"):
        sim_data = {
            'amount': sim_amount,
            'payment_mode': sim_mode,
            'merchant_category': sim_category,
            'dispute_reason_code': sim_reason,
            'ip_shipping_match': int(sim_match),
            'failed_attempts_last_24h': sim_failed,
            'account_age_days': 45,
            'past_dispute_count': sim_disputes,
            'is_vpn_used': int(sim_vpn)
        }
        
        sim_df = pd.DataFrame([sim_data])
        sim_encoded = pd.get_dummies(sim_df, columns=['payment_mode', 'merchant_category', 'dispute_reason_code'], drop_first=True)
        sim_encoded = sim_encoded.reindex(columns=feature_names, fill_value=0)
        
        score = float(model.predict_proba(sim_encoded)[0][1])
        res = route_dispute(score)
        
        st.success(f"Webhook Ingested Successfully! Fraud Score: **{score:.1%}** -> Decision: **{res['decision']}**")
        
        with st.spinner("Generating automated rebuttal..."):
            pkg = assemble_evidence_package(sim_data, list(feature_importance.head(5).index), res['decision'])
            st.json(pkg)

with tab4:
    st.subheader("XGBoost Performance Metrics")
    report = metrics['precision_recall']['1']
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{report['precision']:.2f}")
    c2.metric("Recall", f"{report['recall']:.2f}")
    c3.metric("F1 Score", f"{report['f1-score']:.2f}")

    cm = metrics['confusion_matrix']
    st.write(f"True Negatives: {cm['tn']} | False Positives: {cm['fp']} | False Negatives: {cm['fn']} | True Positives: {cm['tp']}")
    st.metric("Estimated Cost Impact", f"${metrics['total_cost']:,}")