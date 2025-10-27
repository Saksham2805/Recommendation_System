import pandas as pd
import random

# --------------------------
# STEP 1: Load datasets
# --------------------------
netflix_df = pd.read_csv("data/raw/Netflix_Titles.csv")
disney_df = pd.read_csv("data/raw/Disney_Plus_Titles.csv")
prime_df = pd.read_csv("data/raw/Amazon_Prime_Titles.csv")

# --------------------------
# STEP 2: Keep required columns
# --------------------------
cols = ['show_id', 'type', 'title', 'cast', 'duration', 'listed_in', 'description']
netflix_df = netflix_df[cols].dropna()
disney_df = disney_df[cols].dropna()
prime_df = prime_df[cols].dropna()

# --------------------------
# STEP 3: Save cleaned datasets
# --------------------------
netflix_df.to_csv("data/cleaned/Netflix_data.csv", index=False)
disney_df.to_csv("data/cleaned/Disney_Plus_data.csv", index=False)
prime_df.to_csv("data/cleaned/Amazon_Prime_data.csv", index=False)

# --------------------------
# STEP 3: Merge & Save
# --------------------------
# all_data = pd.concat([netflix_df, disney_df, prime_df], ignore_index=True)
# all_data.to_csv("aggregated_all_platforms_data.csv", index=False)

# print("✅ Done aggregating and cleaning data for all platforms!")
# print(all_data.head())
