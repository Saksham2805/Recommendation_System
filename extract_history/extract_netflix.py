# Example (conceptual). You must install selenium and the appropriate webdriver.
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
from dotenv import load_dotenv

# Merge series + season, remove episode info
def merge_series_season(title):
    parts = str(title).split(':')
    if len(parts) >= 2:
        return f"{parts[0].strip()}: {parts[1].strip()}"
    return title.strip()

def get_netflix_full_history(email, password, profile_name, output_dir="histories"):
    driver = webdriver.Chrome()
    driver.get("https://www.netflix.com/login")

    # -------------------------
    # LOGIN
    # -------------------------
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.NAME, "userLoginId"))
    )
    driver.find_element(By.NAME, "userLoginId").send_keys(email)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # -------------------------
    # PROFILE SELECTION
    # -------------------------
    try:
        profile = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"//span[text()='{profile_name}']"))
        )
        profile.click()
        time.sleep(3)
    except:
        print(f"No profile selection screen — using default or already in '{profile_name}'.")

    # -------------------------
    # VIEWING ACTIVITY PAGE
    # -------------------------
    driver.get("https://www.netflix.com/viewingactivity")

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "li.retableRow"))
    )

    # -------------------------
    # LOAD ALL HISTORY
    # -------------------------
    last_count = 0
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Click "Load More" if available
        try:
            load_more = driver.find_element(By.CSS_SELECTOR, "button.btn-blue")
            if load_more.is_displayed():
                driver.execute_script("arguments[0].click();", load_more)
                print("Clicked 'Load More' button...")
                time.sleep(2)
        except:
            pass

        rows = driver.find_elements(By.CSS_SELECTOR, "li.retableRow")
        if len(rows) == last_count:  # no new entries
            break
        last_count = len(rows)
        print(f"Loaded {last_count} rows so far...")

    # -------------------------
    # SCRAPE DATA
    # -------------------------
    data = []
    rows = driver.find_elements(By.CSS_SELECTOR, "li.retableRow")
    for r in rows:
        date = r.find_element(By.CSS_SELECTOR, ".col.date").text.strip()
        title = r.find_element(By.CSS_SELECTOR, ".col.title").text.strip()
        data.append({"date": date, "title": title})

    df = pd.DataFrame(data)
    
    # Group by title and count how many times each appears
    df_summary = df.groupby("title").size().reset_index(name="count")

    df['series_season'] = df_summary['title'].apply(merge_series_season)
    # Keep only unique series_season rows
    df_merged = df.drop_duplicates(subset=['series_season']).reset_index(drop=True)
    # Count occurrences
    final_df = df['series_season'].value_counts().reset_index()
    final_df.columns = ['title', 'count']
    # Optional: sort alphabetically by title
    final_df = final_df.sort_values('title').reset_index(drop=True)


    # Save to CSV
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)
    safe_profile = profile_name.replace(" ", "_").lower()
    file_path = os.path.join(output_dir, f"netflix_history_{safe_profile}.csv")
    final_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"✅ Full history for profile '{profile_name}' saved to {file_path} ({len(final_df)} rows)")

    # -------------------------
    # LOGOUT
    # -------------------------
    try:
        driver.get("https://www.netflix.com/SignOut")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "userLoginId"))
        )
        print("Logged out successfully.")
    except:
        print("Logout may not have completed.")

    driver.quit()
    return df_summary

# -------------------------
# USAGE EXAMPLE
# -------------------------
if __name__ == "__main__":
    load_dotenv()
    EMAIL = os.getenv("NETFLIX_EMAIL")
    PASSWORD = os.getenv("NETFLIX_PASS")
    PROFILE_NAME = os.getenv("NETFLIX_PROFILE")

    df = get_netflix_full_history(EMAIL, PASSWORD, PROFILE_NAME)
    print(df.head(), f"\nTotal entries: {len(df)}")