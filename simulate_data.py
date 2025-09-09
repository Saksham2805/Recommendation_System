import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate a random date
def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

start_date = datetime(2018, 1, 1)
end_date = datetime(2025, 1, 1)

# --------------------------
# STEP 1: Load datasets
# --------------------------
netflix_df = pd.read_csv("data/Netflix_Titles.csv")
disney_df = pd.read_csv("data/Disney_Plus_Titles.csv")
prime_df = pd.read_csv("data/Amazon_Prime_Titles.csv")

# --------------------------
# STEP 2: Keep required columns
# --------------------------
cols = ['show_id', 'title', 'listed_in', 'description']
netflix_df = netflix_df[cols].dropna()
disney_df = disney_df[cols].dropna()
prime_df = prime_df[cols].dropna()

# --------------------------
# STEP 3: Simulate user histories
# --------------------------
def simulate_histories(df, platform_name, num_users=1000, movies_per_user=5):
    users = [f"{platform_name}_user_{i}" for i in range(num_users)]
    histories = []

    for user in users:
        liked_movies = df.sample(movies_per_user)
        for _, row in liked_movies.iterrows():
            histories.append({
                "platform": platform_name,
                "item_id": row['show_id'],
                "title": row['title'],
                "description": str(row['description'])[:500],
                "category": row['listed_in'],
                "user_id": user,
                "interaction_type": "watch",
                "interaction_value": 1,
                "timestamp": random_date(start_date, end_date)
            })
    return histories

# Generate histories
user_histories_netflix = simulate_histories(netflix_df, "netflix")
user_histories_disney = simulate_histories(disney_df, "disney_plus")
user_histories_prime = simulate_histories(prime_df, "amazon_prime")

# --------------------------
# STEP 4: Merge & Save
# --------------------------
all_histories = pd.DataFrame(user_histories_netflix + user_histories_disney + user_histories_prime)
all_histories.to_csv("simulated_user_histories_all_platforms.csv", index=False)

print("✅ Simulated user histories saved for all platforms!")
print(all_histories.head())
