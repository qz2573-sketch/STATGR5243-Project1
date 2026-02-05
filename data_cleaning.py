import pandas as pd
import os

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "data_raw"
OUTPUT_DIR = "data_clean"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# Column Mapping Configuration
# Modify this section based on your analysis of the raw data
# ==========================================
# This dictionary defines how to merge columns with the same meaning but different names
COLUMN_MERGE_MAPPING = {
    # Datetime columns: Yellow uses 'tpep_...', Green uses 'lpep_...'
    'pickup_datetime': {
        'yellow': 'tpep_pickup_datetime',
        'green': 'lpep_pickup_datetime'
    },
    'dropoff_datetime': {
        'yellow': 'tpep_dropoff_datetime',
        'green': 'lpep_dropoff_datetime'
    },
    # Location columns: Standardize case
    'pulocationid': {
        'yellow': 'PULocationID',
        'green': 'PULocationID'
    },
    'dolocationid': {
        'yellow': 'DOLocationID',
        'green': 'DOLocationID'
    }
}

# Note: Only common columns (present in both Yellow and Green) will be kept.
# Type-specific columns (e.g., Airport_fee for Yellow, ehail_fee for Green) will be excluded.

def analyze_raw_data():
    """
    Load and analyze raw data files to identify column differences.
    This function helps identify which columns need to be merged.
    
    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("STEP 1: Analyzing Raw Data Structure")
    print("="*60)
    
    yellow_path = os.path.join(INPUT_DIR, 'yellow_tripdata_2025_sample.csv')
    green_path = os.path.join(INPUT_DIR, 'green_tripdata_2025_sample.csv')
    
    try:
        df_yellow = pd.read_csv(yellow_path)
        df_green = pd.read_csv(green_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    yellow_cols = set(df_yellow.columns)
    green_cols = set(df_green.columns)
    
    common_cols = sorted(yellow_cols & green_cols)
    yellow_only = sorted(yellow_cols - green_cols)
    green_only = sorted(green_cols - yellow_cols)
    
    print(f"\nYellow Taxi: {len(df_yellow)} rows, {len(df_yellow.columns)} columns")
    print(f"Green Taxi: {len(df_green)} rows, {len(df_green.columns)} columns")
    
    print(f"\nCommon columns ({len(common_cols)}):")
    for col in common_cols:
        print(f"  - {col}")
    
    print(f"\nYellow-only columns ({len(yellow_only)}):")
    for col in yellow_only:
        print(f"  - {col}")
        # Show sample values
        sample_vals = df_yellow[col].dropna().head(3).tolist()
        print(f"    Sample values: {sample_vals}")
    
    print(f"\nGreen-only columns ({len(green_only)}):")
    for col in green_only:
        print(f"  - {col}")
        # Show sample values
        sample_vals = df_green[col].dropna().head(3).tolist()
        print(f"    Sample values: {sample_vals}")
    
    # Identify columns that likely have the same meaning
    print("\n" + "="*60)
    print("Column Merging Strategy:")
    print("="*60)
    print("  - tpep_pickup_datetime (Yellow) <-> lpep_pickup_datetime (Green) -> pickup_datetime")
    print("  - tpep_dropoff_datetime (Yellow) <-> lpep_dropoff_datetime (Green) -> dropoff_datetime")
    print("\nNote: Only columns present in BOTH datasets will be kept.")
    print("Type-specific columns (Yellow-only or Green-only) will be excluded.")
    
    return {
        'yellow': df_yellow,
        'green': df_green,
        'common_cols': common_cols,
        'yellow_only': yellow_only,
        'green_only': green_only
    }

def merge_and_standardize_columns(df_yellow, df_green):
    """
    Merge columns with the same meaning but different names across Yellow and Green datasets.
    This function applies the column mapping defined in COLUMN_MERGE_MAPPING.
    
    Args:
        df_yellow: Yellow taxi DataFrame
        df_green: Green taxi DataFrame
    
    Returns:
        Tuple of (standardized_yellow_df, standardized_green_df)
    """
    print("\n" + "="*60)
    print("STEP 2: Merging and Standardizing Columns")
    print("="*60)
    
    # Create copies to avoid modifying originals
    yellow_df = df_yellow.copy()
    green_df = df_green.copy()
    
    # Apply column merging based on COLUMN_MERGE_MAPPING
    for standard_name, type_mapping in COLUMN_MERGE_MAPPING.items():
        # Rename columns in yellow dataset
        if 'yellow' in type_mapping and type_mapping['yellow'] in yellow_df.columns:
            yellow_df.rename(columns={type_mapping['yellow']: standard_name}, inplace=True)
            print(f"  Yellow: {type_mapping['yellow']} -> {standard_name}")
        
        # Rename columns in green dataset
        if 'green' in type_mapping and type_mapping['green'] in green_df.columns:
            green_df.rename(columns={type_mapping['green']: standard_name}, inplace=True)
            print(f"  Green: {type_mapping['green']} -> {standard_name}")
    
    # Standardize location ID column names (case consistency)
    if 'PULocationID' in yellow_df.columns:
        yellow_df.rename(columns={'PULocationID': 'pulocationid'}, inplace=True)
    if 'DOLocationID' in yellow_df.columns:
        yellow_df.rename(columns={'DOLocationID': 'dolocationid'}, inplace=True)
    
    if 'PULocationID' in green_df.columns:
        green_df.rename(columns={'PULocationID': 'pulocationid'}, inplace=True)
    if 'DOLocationID' in green_df.columns:
        green_df.rename(columns={'DOLocationID': 'dolocationid'}, inplace=True)
    
    # Add taxi type identifier
    yellow_df['taxi_type'] = 'yellow'
    green_df['taxi_type'] = 'green'
    
    return yellow_df, green_df

def combine_datasets(df_yellow, df_green):
    """
    Combine Yellow and Green datasets, keeping only columns present in both datasets.
    Type-specific columns (e.g., Airport_fee, ehail_fee, trip_type) will be excluded.
    
    Args:
        df_yellow: Standardized Yellow taxi DataFrame
        df_green: Standardized Green taxi DataFrame
    
    Returns:
        Combined DataFrame with only common columns
    """
    print("\n" + "="*60)
    print("STEP 3: Combining Datasets")
    print("="*60)
    
    # Get only common columns (present in both datasets)
    yellow_cols = set(df_yellow.columns)
    green_cols = set(df_green.columns)
    common_columns = sorted(yellow_cols & green_cols)
    
    # Identify columns that will be excluded
    yellow_only = sorted(yellow_cols - green_cols)
    green_only = sorted(green_cols - yellow_cols)
    
    if yellow_only:
        print(f"  Excluding Yellow-only columns ({len(yellow_only)}): {', '.join(yellow_only)}")
    if green_only:
        print(f"  Excluding Green-only columns ({len(green_only)}): {', '.join(green_only)}")
    
    print(f"\n  Keeping only common columns ({len(common_columns)}):")
    for col in common_columns:
        print(f"    - {col}")
    
    # Select only common columns from each dataset
    df_yellow_common = df_yellow[common_columns].copy()
    df_green_common = df_green[common_columns].copy()
    
    # Combine datasets
    combined_df = pd.concat([df_yellow_common, df_green_common], ignore_index=True)
    print(f"\n  Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    print(f"    - Yellow: {len(df_yellow_common)} rows")
    print(f"    - Green: {len(df_green_common)} rows")
    
    return combined_df

def clean_combined_data(df):
    """
    Performs data cleaning on the merged dataset (Yellow and Green taxis combined).
    Handles datetime conversion, date filtering, numeric inconsistencies, 
    trip duration calculation, and missing value removal (rows with any missing values are dropped).
    
    Args:
        df: Combined DataFrame with Yellow and Green taxi data
    
    Returns:
        Cleaned DataFrame ready for analysis (no missing values)
    """
    print("\n" + "="*60)
    print("STEP 4: Cleaning Combined Dataset")
    print("="*60)
    initial_count = len(df)
    print(f"  Initial rows: {initial_count}")
    
    # 1. Convert datetime strings to datetime objects
    before_dt = len(df)
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    if 'dropoff_datetime' in df.columns:
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    
    # Drop rows where datetime parsing failed
    df.dropna(subset=['pickup_datetime', 'dropoff_datetime'], inplace=True)
    after_dt = len(df)
    if before_dt != after_dt:
        print(f"  After datetime conversion: {after_dt} rows (removed {before_dt - after_dt} rows with invalid dates)")

    # 2. Filter Date Range (Targeting July - Oct 2025)
    # This removes data entry errors (e.g., years 2003, 2099)
    before_date_filter = len(df)
    if 'pickup_datetime' in df.columns:
        valid_date_mask = (df['pickup_datetime'].dt.year == 2025) & \
                          (df['pickup_datetime'].dt.month.isin([7, 8, 9, 10]))
        df = df[valid_date_mask]
    after_date_filter = len(df)
    if before_date_filter != after_date_filter:
        print(f"  After date range filter (2025, months 7-10): {after_date_filter} rows (removed {before_date_filter - after_date_filter} rows)")

    # 3. Handle Numeric Inconsistencies (Negative fares/distance)
    before_fare = len(df)
    if 'fare_amount' in df.columns:
        df = df[df['fare_amount'] >= 0]
    after_fare = len(df)
    if before_fare != after_fare:
        print(f"  After removing negative fares: {after_fare} rows (removed {before_fare - after_fare} rows)")
    
    if 'trip_distance' in df.columns:
        # Distance should be positive (absolute value used to fix negative sensor errors)
        negative_dist_count = (df['trip_distance'] < 0).sum()
        if negative_dist_count > 0:
            print(f"  Fixed {negative_dist_count} rows with negative trip distance (converted to absolute value)")
        df['trip_distance'] = df['trip_distance'].abs()

    # 4. Calculate and Clean Trip Duration
    # Calculate duration in minutes
    if 'pickup_datetime' in df.columns and 'dropoff_datetime' in df.columns:
        df['trip_duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
        
        # Filter invalid durations: Remove <= 0 mins or > 24 hours (1440 mins)
        before_duration = len(df)
        df = df[(df['trip_duration_min'] > 0) & (df['trip_duration_min'] <= 1440)]
        after_duration = len(df)
        if before_duration != after_duration:
            invalid_duration = before_duration - after_duration
            print(f"  After removing invalid trip durations (<=0 or >24h): {after_duration} rows (removed {invalid_duration} rows)")

    # 5. Handle Missing Values (Remove rows with any missing values)
    before_missing = len(df)
    # Drop rows with any missing values in any column
    df.dropna(inplace=True)
    after_missing = len(df)
    if before_missing != after_missing:
        removed_missing = before_missing - after_missing
        print(f"  After removing rows with missing values: {after_missing} rows (removed {removed_missing} rows)")

    final_count = len(df)
    removed_total = initial_count - final_count
    print(f"\n  Final rows: {final_count}")
    print(f"  Total removed: {removed_total} rows ({removed_total/initial_count*100:.2f}%)")
    
    # Format the final dataframe: add Trip ID, reorder columns, remove specified columns
    df = format_final_dataframe(df)
    
    return df

def format_final_dataframe(df):
    """
    Format the final cleaned dataframe:
    1. Add Trip ID column (1, 2, 3, ...) - numeric only, no prefix
    2. Remove RatecodeID, VendorID, source_month, store_and_fwd_flag, source_file
    3. Round trip_duration_min to 2 decimal places
    4. Reorder columns: Trip_ID, pickup_datetime, dropoff_datetime, pulocationid, dolocationid, ...
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        Formatted DataFrame with Trip ID and reordered columns
    """
    print("\n" + "="*60)
    print("Formatting Final Dataframe")
    print("="*60)
    
    df = df.copy()
    
    # 1. Remove specified columns
    columns_to_remove = ['RatecodeID', 'VendorID', 'source_month', 'store_and_fwd_flag', 'source_file']
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Removed column: {col}")
    
    # 2. Round trip_duration_min to 2 decimal places
    if 'trip_duration_min' in df.columns:
        df['trip_duration_min'] = df['trip_duration_min'].round(2)
        print(f"  Rounded trip_duration_min to 2 decimal places")
    
    # 3. Add Trip ID column (numeric only: 1, 2, 3, ...)
    df.insert(0, 'Trip_ID', [i+1 for i in range(len(df))])
    print(f"  Added Trip ID column (1 to {len(df)})")
    
    # 4. Reorder columns
    # Order: Trip_ID, pickup_datetime, dropoff_datetime, pulocationid, dolocationid, ... (other columns)
    priority_cols = ['Trip_ID']
    
    # Add datetime columns
    if 'pickup_datetime' in df.columns:
        priority_cols.append('pickup_datetime')
    if 'dropoff_datetime' in df.columns:
        priority_cols.append('dropoff_datetime')
    
    # Add location columns (4th and 5th positions)
    if 'pulocationid' in df.columns:
        priority_cols.append('pulocationid')
    if 'dolocationid' in df.columns:
        priority_cols.append('dolocationid')
    
    # Get all other columns (excluding priority columns)
    other_cols = [col for col in df.columns if col not in priority_cols]
    # Sort other columns alphabetically for consistency
    other_cols = sorted(other_cols)
    
    # Final column order
    final_cols = priority_cols + other_cols
    
    # Ensure all columns in df are included (in case some are missing)
    final_cols = [col for col in final_cols if col in df.columns]
    
    # Reorder dataframe
    df = df[final_cols]
    
    print(f"  Reordered columns:")
    print(f"    1st: {final_cols[0]}")
    if len(final_cols) >= 2:
        print(f"    2nd: {final_cols[1]}")
    if len(final_cols) >= 3:
        print(f"    3rd: {final_cols[2]}")
    if len(final_cols) >= 4:
        print(f"    4th: {final_cols[3]}")
    if len(final_cols) >= 5:
        print(f"    5th: {final_cols[4]}")
    print(f"  Total columns: {len(final_cols)}")
    
    return df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for the combined dataset, broken down by taxi type.
    
    Args:
        df: Cleaned DataFrame with Yellow and Green taxi data
    
    Returns:
        Dictionary containing summary statistics
    """
    stats = {}
    
    # Overall statistics
    stats['total_trips'] = len(df)
    stats['yellow_trips'] = len(df[df['taxi_type'] == 'yellow'])
    stats['green_trips'] = len(df[df['taxi_type'] == 'green'])
    
    # Calculate statistics by taxi type
    numeric_cols = ['trip_distance', 'fare_amount', 'total_amount', 'trip_duration_min', 'passenger_count']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if available_cols:
        stats['by_type'] = df.groupby('taxi_type')[available_cols].agg(['mean', 'median', 'std']).to_dict()
    
    return stats

def main():
    """
    Main function to analyze, merge, clean, and save Yellow and Green taxi data.
    The process follows these steps:
    1. Analyze raw data structure
    2. Merge and standardize columns
    3. Combine datasets
    4. Clean combined data
    5. Save output and display statistics
    """
    # Step 1: Analyze raw data structure
    analysis = analyze_raw_data()
    if analysis is None:
        print("Failed to load raw data files. Please run the acquisition script first.")
        return
    
    df_yellow_raw = analysis['yellow']
    df_green_raw = analysis['green']
    
    # Step 2: Merge and standardize columns
    df_yellow, df_green = merge_and_standardize_columns(df_yellow_raw, df_green_raw)
    
    # Step 3: Combine datasets
    combined_df = combine_datasets(df_yellow, df_green)
    
    # Step 4: Clean combined data (includes formatting)
    cleaned_df = clean_combined_data(combined_df)
    
    # Step 5: Calculate and display summary statistics
    print("\n" + "="*60)
    print("STEP 5: Summary Statistics by Taxi Type")
    print("="*60)
    stats = calculate_summary_statistics(cleaned_df)
    print(f"Total trips: {stats['total_trips']}")
    print(f"  - Yellow: {stats['yellow_trips']}")
    print(f"  - Green: {stats['green_trips']}")
    
    # Display key metrics by type
    if 'by_type' in stats and cleaned_df is not None:
        print("\nKey Metrics by Taxi Type:")
        for col in ['trip_distance', 'fare_amount', 'total_amount', 'trip_duration_min']:
            if col in cleaned_df.columns:
                print(f"\n{col.replace('_', ' ').title()}:")
                for taxi_type in ['yellow', 'green']:
                    if taxi_type in cleaned_df['taxi_type'].values:
                        subset = cleaned_df[cleaned_df['taxi_type'] == taxi_type][col]
                        print(f"  {taxi_type.capitalize()}: Mean={subset.mean():.2f}, Median={subset.median():.2f}")
    
    # Step 6: Save final output (single combined file)
    output_path = os.path.join(OUTPUT_DIR, 'nyc_taxi_combined_2025_cleaned.csv')
    try:
        cleaned_df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"STEP 6: Final Output Saved")
        print(f"{'='*60}")
        print(f"File saved to: {output_path}")
        print(f"Total records: {len(cleaned_df)}")
        print(f"Total columns: {len(cleaned_df.columns)}")
        print(f"\nColumns: {', '.join(cleaned_df.columns.tolist())}")
    except PermissionError:
        print(f"\nError: Cannot save file. Please close the file if it's open in another program.")
        print(f"Target path: {output_path}")

if __name__ == "__main__":
    main()
