import pandas as pd
import numpy as np

def generate_razorpay_dataset(n_samples=10000, random_state=42):
    np.random.seed(random_state)
    
    payment_modes = ['UPI', 'CREDIT_CARD', 'DEBIT_CARD', 'NETBANKING', 'WALLET']
    mcc_categories = ['ELECTRONICS', 'GAMING', 'RETAIL', 'TRAVEL', 'DIGITAL_SERVICES']
    dispute_reasons = ['10.4_FRAUDULENT_TRANSACTION', '13.1_MERCHANDISE_NOT_RECEIVED', '13.3_NOT_AS_DESCRIBED', '10.5_VISA_EASY_PAYMENT']
    
    data = {
        'transaction_id': [f"pay_{i:08d}" for i in range(n_samples)],
        'amount': np.round(np.random.exponential(scale=1500, size=n_samples) + 100, 2),
        'payment_mode': np.random.choice(payment_modes, size=n_samples, p=[0.50, 0.20, 0.15, 0.10, 0.05]),
        'merchant_category': np.random.choice(mcc_categories, size=n_samples),
        'dispute_reason_code': np.random.choice(dispute_reasons, size=n_samples, p=[0.50, 0.25, 0.15, 0.10]),
        'ip_shipping_match': np.random.choice([1, 0], size=n_samples, p=[0.85, 0.15]),
        'failed_attempts_last_24h': np.random.poisson(lam=0.5, size=n_samples),
        'account_age_days': np.random.randint(1, 1000, size=n_samples),
        'past_dispute_count': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.10, 0.04, 0.01]),
        'is_vpn_used': np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10]),
    }
    
    df = pd.DataFrame(data)
    
    fraud_prob = (
        0.02 
        + 0.25 * (df['ip_shipping_match'] == 0)
        + 0.15 * (df['is_vpn_used'] == 1)
        + 0.20 * (df['failed_attempts_last_24h'] > 2)
        + 0.15 * (df['past_dispute_count'] > 0)
        + 0.10 * (df['payment_mode'] == 'CREDIT_CARD')
        + 0.10 * (df['amount'] > 5000)
    )
    
    fraud_prob = np.clip(fraud_prob, 0, 1)
    df['is_chargeback_fraud'] = (np.random.rand(n_samples) < fraud_prob).astype(int)
    
    df.to_csv('data/razorpay_transactions.csv', index=False)
    print("Generated data/razorpay_transactions.csv successfully!")

if __name__ == "__main__":
    generate_razorpay_dataset()