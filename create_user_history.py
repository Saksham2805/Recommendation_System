import pandas as pd
import numpy as np
from random import sample, randint, choice
from datetime import datetime, timedelta

# -----------------------------
# Step 1: Load datasets
# -----------------------------
amazon_df = pd.read_csv("data/Amazon_Prime_Titles.csv")
netflix_df = pd.read_csv("data/Netflix_Titles.csv")
disney_df = pd.read_csv("data/Disney_Plus_Titles.csv")

platform_dfs = {
    "Amazon": amazon_df,
    "Netflix": netflix_df,
    "Disney": disney_df
}

# Define users and interactions
num_users = 5
interaction_types = ["watch", "rate", "like"]
ratings = [1, 2, 3, 4, 5]

def simulate_user_history(df, platform, num_users=5):
    user_histories = []
    for user_id in range(1, num_users + 1):
        sampled_items = df.sample(n=10, replace=False)
        for _, row in sampled_items.iterrows():
            interaction = choice(interaction_types)
            value = randint(4,5) if interaction == "rate" else np.nan  # high ratings for recommendation
            timestamp = datetime.now() - timedelta(days=randint(0, 365))
            
            user_histories.append({
                "platform": platform,
                "user_id": f"user_{user_id}",
                "show_id": row["show_id"],
                "type": row["type"],
                "title": row["title"],
                "director": row["director"],
                "cast": row["cast"],
                "country": row["country"],
                "date_added": row["date_added"],
                "release_year": row["release_year"],
                "rating": row["rating"],
                "duration": row["duration"],
                "listed_in": row["listed_in"],
                "description": row["description"],
                "interaction_type": interaction,
                "interaction_value": value,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
    return pd.DataFrame(user_histories)

# Simulate histories
amazon_users = simulate_user_history(amazon_df, "Amazon")
netflix_users = simulate_user_history(netflix_df, "Netflix")
disney_users = simulate_user_history(disney_df, "Disney")

# Save to CSV
amazon_users.to_csv("Simulated_Amazon_User_Histories.csv", index=False)
netflix_users.to_csv("Simulated_Netflix_User_Histories.csv", index=False)
disney_users.to_csv("Simulated_Disney_User_Histories.csv", index=False)

print("Simulated user histories created for all platforms.")