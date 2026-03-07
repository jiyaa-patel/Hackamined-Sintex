import pandas as pd
df = pd.read_csv("historical_telemetry.csv", nrows=1, header=None)
headers = df.iloc[0].tolist()
matched = [(i, h) for i, h in enumerate(headers) if 'inverters[0]' in str(h).lower()]
for i, m in matched:
    print(f"Col {i}: {m}")
