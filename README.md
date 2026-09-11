
# 🛡️ Razorpay AI Chargeback Defense System

An automated, end-to-end chargeback defense engine built for payment gateways like *Razorpay. This system combines an **XGBoost machine learning classifier* for real-time risk scoring with *Google Gemini 2.5 Flash* for automated, card-network-compliant dispute evidence package generation.
---
Live Link: https://chargeback-defence-system-tbmce9zoh5hmvqbmpd9w9n.streamlit.app/

---

## 📌 Problem Statement
Merchant chargebacks result in billions of dollars lost annually due to high manual review bottlenecks, short network response windows (T+3 days), and strict network reason codes (Visa, Mastercard, RuPay, UPI). 

This project solves chargeback friction by:
1. *Automating Threat Triage:* Instant risk classification into AUTO_SUBMIT_EVIDENCE, ESCALATE_TO_HUMAN, or LIKELY_LEGITIMATE_DISPUTE.
2. *Generative Evidence Assembly:* Automatically drafting structured, network-compliant representment packages using LLMs.
3. *Real-Time Gateway Integration:* Simulating Razorpay payment.dispute.created HTTP webhooks.

---

## 🚀 Key Features

- *XGBoost Risk Engine:* Classifies transactions based on payment mode (UPI, Credit Card, NetBanking, Wallets), VPN detection, network reason codes, and historical merchant disputes.
- *Cost-Matrix Optimization:* Minimizes financial risk by weighing False Positive costs ($5) against False Negative dispute losses ($500).
- *Google Gemini Integration:* Dynamically generates structured JSON rebuttal packages with zero hallucination via pydantic output schema enforcement.
- *Interactive Streamlit Dashboard:* Multi-tab analytical suite featuring triage charts, dispute lookup desks, model performance metrics, and a *Live Webhook Simulator*.

---


## 🛠️ Tech Stack

- *Frontend / Dashboard:* Streamlit
- *Machine Learning:* XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
- *Generative AI:* Google GenAI SDK (gemini-3.6-flash), Pydantic
- *Environment Management:* Python-Dotenv

---

## 📂 Project Structure


chargeback-defence-system/
├── app/
│   └── streamlit_app.py        # Streamlit dashboard & Webhook simulator UI
├── data/
│   ├── razorpay_transactions.csv  # Synthetic domain-specific dataset
│   └── test_sample.csv            # Unencoded holdout sample for evaluation
├── models/
│   ├── model.pkl               # Saved XGBoost model artifact
│   ├── feature_names.pkl       # Encodings schema mapping
│   ├── feature_importance.csv  # Top risk drivers
│   └── metrics.pkl             # Evaluation and cost matrix metrics
├── src/
│   ├── generate_data.py        # Dataset generator (UPI, Card, MCC, Reason Codes)
│   ├── train_model.py          # XGBoost training & preprocessing pipeline
│   ├── escalation.py           # Threshold routing logic
│   └── llm_responder.py        # Gemini structured response engine
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies

---

## ⚙️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/chargeback-defence-system.git](https://github.com/YOUR_GITHUB_USERNAME/chargeback-defence-system.git)
cd chargeback-defence-system

2. Create and Activate Virtual Environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Secrets
Create a .env file in the root folder and add your Google Gemini API key:
GEMINI_API_KEY="your_actual_gemini_api_key_here"

🏃 Running the Application
 * Generate Synthetic Razorpay Dataset:
   python src/generate_data.py

 * Train XGBoost Classifier:
   python src/train_model.py

 * Launch Streamlit Dashboard:
   streamlit run app/streamlit_app.py

📊 Model Evaluation Summary
 * Class Imbalance Handling: Weighted scale optimization (scale_pos_weight).
 * Precision (Fraud Class): ~0.91
 * Recall (Fraud Class): ~0.82
 * Cost Minimization: Reduces manual review labor costs while maximizing merchant representment win-rates.
