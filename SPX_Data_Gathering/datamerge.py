import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# CONFIGURATION — edit these paths as needed
# ─────────────────────────────────────────────

# Path to the sentiment file
SENTIMENT_FILE = "spx_returns_sentiment_merged.csv"

# Folder containing all cleanerSPXData_YYYY.csv files
INPUT_FOLDER = "."

# Folder where merged files will be saved (created if it doesn't exist)
OUTPUT_FOLDER = "merged_output"

# ─────────────────────────────────────────────


def merge_sentiment(sentiment_file, input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load sentiment data once
    print(f"Loading sentiment file: {sentiment_file}")
    sentiment = pd.read_csv(sentiment_file)
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment = sentiment.rename(columns={"date": "Trade Date"})
    print(f"  → {len(sentiment)} sentiment rows loaded\n")

    # Find all matching cleanerSPXData files
    pattern = os.path.join(input_folder, "cleanerSPXData_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching 'cleanerSPXData_*.csv' found in: {input_folder}")
        return

    print(f"Found {len(files)} file(s) to process:\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_folder, filename.replace(".csv", "_merged.csv"))

        print(f"Processing: {filename}")

        # Load the SPX data file
        df = pd.read_csv(filepath, low_memory=False)
        df["Trade Date"] = pd.to_datetime(df["Trade Date"])

        # Left join — keeps all SPX rows, appends sentiment columns where date matches
        merged = df.merge(sentiment, on="Trade Date", how="left")

        # Save result
        merged.to_csv(output_path, index=False)

        # Report
        total = len(merged)
        matched = merged["daily_sentiment"].notna().sum()
        unmatched = total - matched
        print(f"  → {total:,} rows total | {matched:,} matched ({matched/total*100:.1f}%) | {unmatched:,} unmatched")
        print(f"  → Saved to: {output_path}\n")

    print("All done!")


if __name__ == "__main__":
    merge_sentiment(SENTIMENT_FILE, INPUT_FOLDER, OUTPUT_FOLDER)
    print("done")