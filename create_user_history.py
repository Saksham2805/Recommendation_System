import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_user_histories(netflix_file, prime_file, disney_file, num_users=10):
    """
    Generate user history CSV files from Netflix, Amazon Prime, and Disney Plus datasets
    
    Parameters:
    netflix_file (str): Path to Netflix CSV file
    prime_file (str): Path to Amazon Prime CSV file  
    disney_file (str): Path to Disney Plus CSV file
    num_users (int): Number of user history files to generate
    """
    
    # Read the datasets
    try:
        netflix_df = pd.read_csv(netflix_file)
        prime_df = pd.read_csv(prime_file)
        disney_df = pd.read_csv(disney_file)
        
        print(f"Loaded datasets:")
        print(f"Netflix: {len(netflix_df)} entries")
        print(f"Amazon Prime: {len(prime_df)} entries")
        print(f"Disney Plus: {len(disney_df)} entries")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Add platform identifier to each dataset
    netflix_df['platform'] = 'Netflix'
    prime_df['platform'] = 'Amazon Prime'
    disney_df['platform'] = 'Disney Plus'
    
    # Define columns to keep for user history
    columns_to_keep = [
        'show_id', 'type', 'title', 'director', 'cast', 
        'release_year', 'rating', 'duration', 'listed_in', 'platform'
    ]
    
    # Filter columns (keep only existing ones)
    def filter_existing_columns(df, columns_list):
        existing_cols = [col for col in columns_list if col in df.columns]
        return df[existing_cols].copy()
    
    netflix_filtered = filter_existing_columns(netflix_df, columns_to_keep)
    prime_filtered = filter_existing_columns(prime_df, columns_to_keep)
    disney_filtered = filter_existing_columns(disney_df, columns_to_keep)
    
    # Create output directory
    output_dir = 'user_histories'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate user histories
    for user_id in range(1, num_users + 1):
        user_history = []
        
        # Sample from each platform (15-20 movies/shows each)
        for df, platform_name in [(netflix_filtered, 'Netflix'), 
                                  (prime_filtered, 'Amazon Prime'), 
                                  (disney_filtered, 'Disney Plus')]:
            
            if len(df) == 0:
                print(f"Warning: {platform_name} dataset is empty, skipping...")
                continue
                
            # Random sample size between 15-20
            sample_size = random.randint(10, min(20, len(df)))
            
            # Sample random entries (create a valid random seed)
            seed = abs(hash(f"{user_id}_{platform_name}")) % (2**32 - 1)
            platform_sample = df.sample(n=sample_size, random_state=seed)
            user_history.append(platform_sample)
        
        # Combine all platform data for this user
        if user_history:
            user_df = pd.concat(user_history, ignore_index=True)
            
            # Add user-specific columns
            user_df['user_id'] = f'user_{user_id:02d}'
            
            # Add user rating (1-5 stars, with some movies unrated)
            user_ratings = []
            for _ in range(len(user_df)):
                # 20% chance of no rating (None)
                if random.random() < 0.2:
                    user_ratings.append(None)
                else:
                    # Rating between 1-5
                    user_ratings.append(round(random.uniform(1.0, 5.0), 1))
            
            user_df['user_rating'] = user_ratings
            
            # Add watch date (random dates in the last 2 years)
            start_date = datetime.now() - timedelta(days=730)
            watch_dates = []
            for _ in range(len(user_df)):
                random_days = random.randint(0, 730)
                watch_date = start_date + timedelta(days=random_days)
                watch_dates.append(watch_date.strftime('%Y-%m-%d'))
            
            user_df['watch_date'] = watch_dates
            
            # Reorder columns for better readability
            column_order = ['user_id', 'platform', 'show_id', 'title', 'type', 'director', 
                           'cast', 'release_year', 'rating', 'duration', 'listed_in',
                           'user_rating', 'watch_date']
            
            # Keep only existing columns in the desired order
            final_columns = [col for col in column_order if col in user_df.columns]
            user_df = user_df[final_columns]
            
            # Shuffle the rows to mix platforms
            user_df = user_df.sample(frac=1, random_state=user_id*42).reset_index(drop=True)
            
            # Save to CSV
            output_file = f'{output_dir}/user_{user_id:02d}_history.csv'
            user_df.to_csv(output_file, index=False)
            
            print(f"Generated {output_file}: {len(user_df)} entries")
            print(f"  - Netflix: {len(user_df[user_df['platform'] == 'Netflix'])}")
            print(f"  - Amazon Prime: {len(user_df[user_df['platform'] == 'Amazon Prime'])}")
            print(f"  - Disney Plus: {len(user_df[user_df['platform'] == 'Disney Plus'])}")
            print(f"  - Rated movies: {len(user_df[user_df['user_rating'].notna()])}")
            print(f"  - Unrated movies: {len(user_df[user_df['user_rating'].isna()])}")
            print()

def analyze_generated_data(output_dir='user_histories'):
    """
    Analyze the generated user history data
    """
    if not os.path.exists(output_dir):
        print("No user histories directory found!")
        return
    
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in user_histories directory!")
        return
    
    print(f"\n=== Analysis of Generated User Histories ===")
    print(f"Total user history files: {len(csv_files)}")
    
    total_entries = 0
    platform_counts = {'Netflix': 0, 'Amazon Prime': 0, 'Disney Plus': 0}
    total_rated = 0
    total_unrated = 0
    
    for file in csv_files:
        df = pd.read_csv(f"{output_dir}/{file}")
        total_entries += len(df)
        
        for platform in platform_counts.keys():
            platform_counts[platform] += len(df[df['platform'] == platform])
        
        total_rated += len(df[df['user_rating'].notna()])
        total_unrated += len(df[df['user_rating'].isna()])
    
    print(f"Total entries across all users: {total_entries}")
    print(f"Platform distribution:")
    for platform, count in platform_counts.items():
        print(f"  - {platform}: {count} entries")
    
    print(f"Rating distribution:")
    print(f"  - Rated entries: {total_rated}")
    print(f"  - Unrated entries: {total_unrated}")
    print(f"  - Rating percentage: {total_rated/total_entries*100:.1f}%")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    netflix_csv = "data/raw/Netflix_Titles.csv"  # Replace with your Netflix CSV file path
    prime_csv = "data/raw/Amazon_Prime_Titles.csv"  # Replace with your Amazon Prime CSV file path
    disney_csv = "data/raw/Disney_Plus_Titles.csv"  # Replace with your Disney Plus CSV file path
    
    print("Starting user history generation...")
    print("Make sure your CSV files are in the same directory as this script")
    print("or update the file paths below:\n")
    
    # Generate user histories
    generate_user_histories(
        netflix_file=netflix_csv,
        prime_file=prime_csv, 
        disney_file=disney_csv,
        num_users=10
    )
    
    # Analyze the generated data
    analyze_generated_data()
    
    print("\nUser history generation completed!")
    print("Check the 'user_histories' directory for the generated CSV files.")