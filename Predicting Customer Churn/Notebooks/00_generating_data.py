# Generating own Synthetic Data
import numpy as np 
import pandas as pd

def synth_data(n=10000, random_seed=42):
    np.random.seed(random_seed)

    # 1. Generate basic user info
    customer_id = np.arange(1, n+1)

    # Age: Uniform distribution from 18 to 70
    age = np.random.randint(18, 71, size=n)

    # Gender: Male/Female with ~50/50 distribution 
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.5, 0.5])

    # Subscription Tier: Categories with different distributions 
    tiers = ["Basic", "Standard", "Premium"]
    subscription_tier = np.random.choice(tiers, size=n, p=[0.4, 0.4, 0.2])

    # 2. User patterns
    # Average monthly usage count
    usage_count = np.random.poisson(lam=15, size=n)  # Around 15 uses per month

    # Days since last login: Heavily skewed, with most people recently active, but a small fraction absent for >30 days
    last_login_days_ago = np.random.choice(
    [i for i in range(61)],
    size=n,
    p=[
        0.2 if i == 0 
        else 0.01 if i > 30 
        else 0.5 / 30 
        for i in range(61)
    ]
)

    # Payment history: # of late payments in last 6 months 
    late_payments_6m = np.random.poisson(lam=0.2, size=n)

    # Monthly spend: Correlated with subscription tier: Basic: mean $20, standard: mean $50, premium: mean $100
    spend_means={"Basic": 20, "Standard": 50, "Premium": 100}
    monthly_spend = np.array([np.random.normal(spend_means[tier], 5) for tier in subscription_tier])
    monthly_spend = np.round(np.clip(monthly_spend, 0, 200), 2)  # Clip at 0 & 200, round to cents

    # Region / location: Weighted for demostration 
    locations = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    location_probs = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
    region = np.random.choice(locations, size=n, p=location_probs)

    # 3. Underlying logistic model for churn probability
    # I define churn odds as a function of some variables plus some random noise
    alpha = -2.0 
    
    # Numeric encoding for supscription tier in the logistic model
    tier_map = {"Basic": 0, "Standard": 1, "Premium": 2}
    tier_num = np.array([tier_map[t] for t in subscription_tier])

    # Construct the log-odds
    log_odds = (
        alpha
        + 0.05 * (age - 30)
        - 1    * (tier_num == 2).astype(float)    # Premium user => reduce churn
        + 0.05 * last_login_days_ago
        + 0.7  * late_payments_6m
        - 0.02 * monthly_spend 
        - 0.3  * (tier_num == 2).astype(float) * monthly_spend
        + np.random.normal(0, 0.3, size=n)
    )

    # Probability of churn 
    churn_prob = 1 / (1 + np.exp(-log_odds))

    # Deterermin churn: 1 if random draw < churn_prob, else 0
    churn = (np.random.rand(n) < churn_prob).astype(int)

    # Assembnle data into a DF 
    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "subscription_tier": subscription_tier,
        "usage_count": usage_count,
        "last_login_days_ago": last_login_days_ago,
        "late_payments_6m": late_payments_6m,
        "monthly_spend": monthly_spend,
        "region": region, 
        "churn": churn
    })

    return df

if __name__ == "__main__":
    df = synth_data(n=10000, random_seed=42)
    df.to_csv(r"C:\Users\sebas\OneDrive - BI Norwegian Business School (BIEDU)\GitHub\Portfolio\Predicting Customer Churn\Data\synthetic_churn_data.csv", index=False)
    print("Synthetic churn dataset generated and saved!")
