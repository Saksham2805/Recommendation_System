from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
from dotenv import load_dotenv

def get_amazon_prime_history(email, password, profile_name, output_file="prime_watch_history.csv"):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        # 1. Go to login page
        driver.get("https://www.primevideo.com/auth-redirect/ref=av_auth_return_redir")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "ap_email")))

        # 2. Login
        driver.find_element(By.ID, "ap_email").send_keys(email)
        driver.find_element(By.ID, "continue").click()
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "ap_password")))
        driver.find_element(By.ID, "ap_password").send_keys(password)
        driver.find_element(By.ID, "signInSubmit").click()

        # 2.5. Profile Selection - FIXED VERSION
        time.sleep(5)
        profile_selected = False
        
        try:
            print("Looking for profile dropdown...")
            
            # First, try to click on the profile dropdown trigger
            profile_trigger_selectors = [
                "button[data-testid='pv-nav-account-and-profiles-dropdown-trigger']",
                "button[aria-label*='Profile']",
                ".ODY5qo.oX9EOY._1Oqfgg"
            ]
            
            profile_trigger = None
            for selector in profile_trigger_selectors:
                try:
                    profile_trigger = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    print(f"✅ Found profile trigger with selector: {selector}")
                    break
                except Exception as e:
                    print(f"Profile trigger selector {selector} failed: {e}")
                    continue
            
            if profile_trigger:
                # Get current profile name from the active profile element
                try:
                    current_profile_elem = driver.find_element(By.CSS_SELECTOR, "span[data-testid*='active-profile-']")
                    current_profile_testid = current_profile_elem.get_attribute("data-testid")
                    current_profile_name = current_profile_testid.replace("active-profile-", "") if current_profile_testid else ""
                    print(f"Current active profile: {current_profile_name}")
                    
                    # If no profile_name specified or current profile matches, continue
                    if not profile_name or current_profile_name.lower() == profile_name.lower():
                        print(f"✅ Already on correct profile: {current_profile_name}")
                        profile_selected = True
                    else:
                        # Need to switch profile
                        print(f"Need to switch from '{current_profile_name}' to '{profile_name}'")
                        
                        # Click the dropdown trigger
                        profile_trigger.click()
                        time.sleep(2)
                        
                        # Wait for dropdown to appear
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='pv-nav-account-and-profiles-dropdown']"))
                        )
                        
                        # Look for profile buttons in the dropdown
                        profile_buttons = driver.find_elements(By.CSS_SELECTOR, "button[data-testid*='pv-nav-change-profile-']")
                        
                        if profile_buttons:
                            print(f"Found {len(profile_buttons)} profile options")
                            
                            profile_found = False
                            for button in profile_buttons:
                                try:
                                    # Get the profile name from the button text or aria-label
                                    button_text = button.get_attribute("aria-label") or button.text.strip()
                                    print(f"Profile option: {button_text}")
                                    
                                    # Case-insensitive comparison
                                    if button_text.lower() == profile_name.lower():
                                        print(f"✅ Found matching profile: {button_text}")
                                        
                                        # Scroll to button if needed
                                        driver.execute_script("arguments[0].scrollIntoView(true);", button)
                                        time.sleep(1)
                                        
                                        # Click the profile button
                                        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(button))
                                        button.click()
                                        print(f"Clicked on profile: {button_text}")
                                        
                                        # Wait for page to reload/switch profile
                                        time.sleep(5)
                                        
                                        # Verify profile switch was successful
                                        try:
                                            new_profile_elem = WebDriverWait(driver, 10).until(
                                                EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-testid*='active-profile-']"))
                                            )
                                            new_profile_testid = new_profile_elem.get_attribute("data-testid")
                                            new_profile_name = new_profile_testid.replace("active-profile-", "") if new_profile_testid else ""
                                            
                                            if new_profile_name.lower() == profile_name.lower():
                                                print(f"✅ Successfully switched to profile: {new_profile_name}")
                                                profile_selected = True
                                            else:
                                                print(f"⚠️ Profile switch may not have worked. Current: {new_profile_name}")
                                        except Exception as ve:
                                            print(f"Could not verify profile switch: {ve}")
                                        
                                        profile_found = True
                                        break
                                        
                                except Exception as be:
                                    print(f"Error processing profile button: {be}")
                                    continue
                            
                            if not profile_found:
                                print(f"❌ Profile '{profile_name}' not found in available options")
                                # Click somewhere else to close dropdown
                                driver.find_element(By.TAG_NAME, "body").click()
                        else:
                            print("No profile buttons found in dropdown")
                            
                except Exception as pe:
                    print(f"Error getting current profile info: {pe}")
            
            if not profile_selected and profile_name:
                print(f"⚠️ Could not switch to profile '{profile_name}' - continuing with current profile")
            elif not profile_name:
                print("No specific profile requested - continuing with current profile")
                profile_selected = True
                
        except Exception as e:
            print("Profile selection error occurred:", str(e))
            # Continue anyway as some accounts might not have profile selection
        
        # 3. Go to Watch History
        time.sleep(5)
        driver.get("https://www.primevideo.com/settings/watch-history")
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.Wiv2s6[data-testid='activity-history-item']"))
        )

        # 4. Scroll until end of page
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # let new items load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # 5. Extract titles
        items = driver.find_elements(By.CSS_SELECTOR, "div.Wiv2s6[data-testid='activity-history-item']")
        data = []
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, "a._1NNx6V").text
            except:
                title = "N/A"

            try:
                episodes = item.find_element(By.CSS_SELECTOR, "div.H1e4O2 label").text
            except:
                episodes = "N/A"

            data.append({"title": title, "episodes_watched": episodes})

        # 6. Save to CSV
        df = pd.DataFrame(data)

        # Group by title and count how many times each appears
        df_summary = df.groupby("title").size().reset_index(name="count")

        # Save to CSV
        df_summary.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ Extracted {len(df_summary)} unique titles. Saved to {output_file}")

        return df_summary

    finally:
        # -------------------------
        # LOGOUT
        # -------------------------
        try:
            # 7. Log out
            driver.get("https://www.primevideo.com")  # go to home page
            time.sleep(2)
            
            # Click on profile dropdown trigger
            account_menu = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='pv-nav-account-and-profiles-dropdown-trigger']"))
            )
            account_menu.click()
            time.sleep(1)
            
            # Click Sign Out
            sign_out = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-testid='pv-nav-sign-out']"))
            )
            sign_out.click()
            print("✅ Logged out successfully")
        except Exception as logout_error:
            print(f"⚠️ Could not log out automatically: {logout_error}")

        time.sleep(2)
        driver.quit()
# -------------------------
# USAGE EXAMPLE
# -------------------------
if __name__ == "__main__":
    load_dotenv()
    EMAIL = os.getenv("AMAZON_PRIME_EMAIL")
    PASSWORD = os.getenv("AMAZON_PRIME_PASS")
    PROFILE = os.getenv("AMAZON_PRIME_PROFILE")

    print("🔍 Amazon Prime History Debug Scraper")
    print("This will create detailed debug files to help troubleshoot the scraping process")
    print("="*80)
    
    # Run the debug scraper
    safe_profile = PROFILE.replace(" ", "_").lower()
    output_dir = "histories"
    file_path = os.path.join(output_dir, f"prime_history_{safe_profile}.csv")
    result_df = get_amazon_prime_history(EMAIL, PASSWORD, PROFILE, file_path)
    
    if not result_df.empty:
        print(f"\n🎉 SUCCESS! Found {len(result_df)} unique titles")
        print(result_df.head())
    else:
        print(f"\n❌ FAILED! No data extracted")
        print("Check the debug files created for troubleshooting:")
        print("- Screenshots: debug_*.png")
        print("- Page sources: debug_*.html")
        print("- Selector results: debug_results_*.json")