import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import os

# ==========================================
# Configuration
# ==========================================
BASE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
TARGET_YEAR = 2025
TARGET_MONTHS = [7, 8, 9, 10]

# Modification: Only keep 'yellow' and 'green'
TARGET_TYPES = ['yellow', 'green'] 
SAMPLE_SIZE_PER_MONTH = 200  # 200 rows per month per vehicle type

OUTPUT_DIR = "data_raw"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_data_links(url):
    """
    Scrapes the provided URL to find all parquet file links.
    """
    print(f"Scraping page for links: {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract links ending in .parquet
        links = [a['href'].strip() for a in soup.find_all('a', href=True) if a['href'].strip().endswith('.parquet')]
        return links
    except Exception as e:
        print(f"Failed to scrape page: {e}")
        return []

def filter_links(links, vehicle_type, year, months):
    """
    Filters links based on vehicle type, year, and target months.
    Returns a dictionary mapping month to link.
    """
    filtered = {}
    for link in links:
        for month in months:
            month_str = f"{month:02d}"
            # Match format like: yellow_tripdata_2025-07
            if f"{vehicle_type}_tripdata_{year}-{month_str}" in link:
                filtered[month] = link
                break  # Each month should have only one link
    return filtered

def download_and_sample(links_dict, vehicle_type, sample_size_per_month):
    """
    Downloads files for each month, reads them into memory, and samples exactly 
    sample_size_per_month rows from each month.
    
    Args:
        links_dict: Dictionary mapping month number to download link
        vehicle_type: Type of vehicle ('yellow' or 'green')
        sample_size_per_month: Number of rows to sample from each month
    
    Returns:
        Combined DataFrame with samples from all months
    """
    if not links_dict:
        return None
    
    all_samples = []
    month_names = {7: 'July', 8: 'August', 9: 'September', 10: 'October'}
    
    print(f"Processing {vehicle_type} (Found {len(links_dict)} months)...")
    
    # Process each month separately
    for month, link in sorted(links_dict.items()):
        try:
            month_name = month_names.get(month, f"Month {month}")
            print(f"  Downloading {month_name} ({month}/2025): {link}")
            r = requests.get(link)
            r.raise_for_status()
            f = io.BytesIO(r.content)
            
            # Read Parquet file
            df = pd.read_parquet(f)
            print(f"    Loaded {len(df):,} rows from {month_name}")
            
            # Sample exactly sample_size_per_month rows from this month
            if len(df) >= sample_size_per_month:
                sample_df = df.sample(n=sample_size_per_month, random_state=42)
                print(f"    Sampled {len(sample_df)} rows from {month_name}")
            else:
                sample_df = df
                print(f"    Warning: Only {len(df)} rows available in {month_name}, using all rows")
            
            # Add month metadata for tracking
            sample_df['source_month'] = month
            sample_df['source_file'] = link.split('/')[-1]
            
            all_samples.append(sample_df)
        except Exception as e:
            print(f"  Error processing {month_name}: {e}")
            continue

    if all_samples:
        # Concatenate all monthly samples
        final_df = pd.concat(all_samples, ignore_index=True)
        print(f"\n  Total combined: {len(final_df)} rows from {len(all_samples)} months")
        return final_df
    return None

def main():
    """
    Main function to acquire NYC taxi data.
    For each vehicle type (Yellow and Green), downloads data for each target month
    (July, August, September, October 2025) and samples 200 rows per month.
    Final output: 800 rows per vehicle type (4 months × 200 rows).
    """
    # 1. Get all available links
    print("="*60)
    print("NYC Taxi Data Acquisition")
    print("="*60)
    print(f"Target: {len(TARGET_TYPES)} vehicle types × {len(TARGET_MONTHS)} months × {SAMPLE_SIZE_PER_MONTH} rows/month")
    print(f"Expected total: {len(TARGET_TYPES)} types × {len(TARGET_MONTHS)} months × {SAMPLE_SIZE_PER_MONTH} = {len(TARGET_TYPES) * len(TARGET_MONTHS) * SAMPLE_SIZE_PER_MONTH} rows per type")
    print()
    
    all_links = get_data_links(BASE_URL)
    print(f"Found {len(all_links)} total parquet file links\n")
    
    # 2. Iterate through Yellow and Green types
    for v_type in TARGET_TYPES:
        print(f"\n{'='*60}")
        print(f"Processing Vehicle Type: {v_type.upper()}")
        print(f"{'='*60}")
        
        # Filter links by month (returns dict: month -> link)
        target_links = filter_links(all_links, v_type, TARGET_YEAR, TARGET_MONTHS)
        
        if not target_links:
            print(f"No data found for {v_type}.")
            continue
        
        # Download and sample 200 rows from each month
        df_sampled = download_and_sample(target_links, v_type, SAMPLE_SIZE_PER_MONTH)
        
        if df_sampled is not None:
            output_file = os.path.join(OUTPUT_DIR, f"{v_type}_tripdata_2025_sample.csv")
            df_sampled.to_csv(output_file, index=False)
            print(f"\n✓ Successfully saved: {output_file}")
            print(f"  Total rows: {len(df_sampled)}")
            print(f"  Expected: {len(TARGET_MONTHS)} months × {SAMPLE_SIZE_PER_MONTH} = {len(TARGET_MONTHS) * SAMPLE_SIZE_PER_MONTH} rows")
            
            # Show breakdown by month
            if 'source_month' in df_sampled.columns:
                month_counts = df_sampled['source_month'].value_counts().sort_index()
                print(f"  Breakdown by month:")
                month_names = {7: 'July', 8: 'August', 9: 'September', 10: 'October'}
                for month, count in month_counts.items():
                    month_name = month_names.get(month, f"Month {month}")
                    print(f"    {month_name}: {count} rows")

if __name__ == "__main__":
    main()