import pandas as pd

def analyze_csv():
    # Read just the top 2 rows to get the structure without loading 455MB into memory
    df_sample = pd.read_csv("historical_telemetry.csv", nrows=2, header=None)
    
    # Print the raw content to understand how columns map
    print("CSV Shape:", df_sample.shape)
    
    print("\n--- Row 1 (Potential Headers or Values) ---")
    for i, val in enumerate(df_sample.iloc[0]):
        print(f"Col {i}: {val}")
        
    print("\n--- Row 2 (Values) ---")
    for i, val in enumerate(df_sample.iloc[1]):
        print(f"Col {i}: {val}")

if __name__ == "__main__":
    analyze_csv()
