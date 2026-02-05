import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import os
import re

# ==========================================
# Configuration
# ==========================================
BASE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
TARGET_YEAR = 2025
# Months: July, August, September, October
TARGET_MONTHS = [7, 8, 9, 10]
# Vehicle types to acquire
TARGET_TYPES = ['yellow', 'green', 'fhv'] 
# Required sample size per vehicle type
SAMPLE_SIZE_PER_TYPE = 200 

# Directory to save the processed datasets
OUTPUT_DIR = "data_raw"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_data_links(url):
    """
    Step 1: Web Scraping - Extract all Parquet file links from the TLC website.
    This fulfills the project requirement for acquiring data via web scraping.
    """
    print(f"Scraping page for links: {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        # Find all anchor tags with href attributes
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            href = href.strip()
            # We are interested in .parquet files (the current TLC format)
            if href.endswith('.parquet'):
                links.append(href)
        
        print(f"Successfully found {len(links)} file links.")
        return links
    except Exception as e:
        print(f"Failed to scrape page: {e}")
        return []

def filter_links(links, vehicle_type, year, months):
    """
    Step 2: Filter links to match the specific year, months, and vehicle type.
    """
    filtered = []
    for link in links:
        # Expected format example: yellow_tripdata_2025-07.parquet
        for month in months:
            # Format month as two digits (e.g., 7 -> 07)
            month_str = f"{month:02d}"
            # Construct the search string based on file naming convention
            search_str = f"{vehicle_type}_tripdata_{year}-{month_str}"
            
            if search_str in link:
                filtered.append(link)
    return filtered

def download_and_sample(links, vehicle_type, total_sample_size):
    """
    Step 3 & 4: Download the specific datasets and perform random sampling.
    This handles the 'Data Acquisition' and prepares for 'Handling Inconsistencies'
    by reducing data size for easier analysis.
    """
    if not links:
        print(f"Warning: No download links found for {vehicle_type} in the specified months.")
        return None

    # Calculate how many samples to take per file to reach the total target
    # Note: We take slightly more to ensure we have enough after filtering
    all_samples = []
    
    print(f"Processing {vehicle_type} data (Found {len(links)} files)...")
    
    for idx, link in enumerate(links):
        try:
            print(f"  Downloading and reading: {link}")
            
            # Stream the download to memory (avoiding saving massive files to disk)
            r = requests.get(link)
            f = io.BytesIO(r.content)
            
            # Read the Parquet file into a pandas DataFrame
            df = pd.read_parquet(f)
            
            # Sample data from this specific month
            # We take a larger chunk initially to ensure we have enough to randomize later
            n_sample = 100 
            
            if len(df) > n_sample:
                sample_df = df.sample(n=n_sample, random_state=42)
            else:
                sample_df = df 
            
            # Add metadata for tracking source (Feature Engineering preparation)
            filename = link.split('/')[-1]
            sample_df['source_file'] = filename
            
            all_samples.append(sample_df)
            
        except Exception as e:
            print(f"  Error processing file {link}: {e}")
            continue

    if all_samples:
        # Concatenate all monthly samples into one DataFrame
        combined_df = pd.concat(all_samples, ignore_index=True)
        
        # Perform the final random sample to get exactly the requested amount (200)
        if len(combined_df) > total_sample_size:
            final_df = combined_df.sample(n=total_sample_size, random_state=42)
        else:
            final_df = combined_df
            
        return final_df
    else:
        return None

def main():
    # 1. Scrape all available links
    all_links = get_data_links(BASE_URL)
    
    # 2. Iterate through each vehicle type
    for v_type in TARGET_TYPES:
        print(f"\n--- Processing Vehicle Type: {v_type} ---")
        
        # Filter for the specific months in 2025
        target_links = filter_links(all_links, v_type, TARGET_YEAR, TARGET_MONTHS)
        
        if not target_links:
            print(f"No data found for {v_type} in {TARGET_YEAR} (Months 7-10).")
            continue
            
        # Download, process, and sample the data
        df_sampled = download_and_sample(target_links, v_type, SAMPLE_SIZE_PER_TYPE)
        
        if df_sampled is not None:
            # 3. Save the sampled dataset to CSV
            # This file serves as the raw dataset for the project deliverables
            output_file = os.path.join(OUTPUT_DIR, f"{v_type}_tripdata_2025_sample.csv")
            df_sampled.to_csv(output_file, index=False)
            print(f"Success! {v_type} data saved to: {output_file} (Rows: {len(df_sampled)})")
            
            # Brief EDA (Exploratory Data Analysis) preview
            print("Data Preview:")
            print(df_sampled.head(2))
        else:
            print(f"Failed to acquire data for {v_type}.")

if __name__ == "__main__":
    main()