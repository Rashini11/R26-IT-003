import pandas as pd

# Step 1: Read file as single column
df = pd.read_csv("data/raw/sensor/wind_wave.csv", header=None)

# Step 2: Split by semicolon
df = df[0].str.split(";", expand=True)

# Step 3: Set first row as header
df.columns = df.iloc[0]

# Step 4: Remove header row
df = df[1:]

# Step 5: Clean column names
df.columns = df.columns.str.replace('"', '').str.strip()

print("Columns:", df.columns)

# Step 6: Convert to numeric
df['sigheight'] = pd.to_numeric(df['sigheight'], errors='coerce')
df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce')

# Step 7: Remove missing / invalid values
df = df.dropna(subset=['sigheight', 'windspeed'])
df = df[(df['sigheight'] != -99.9) & (df['windspeed'] != -99.9)]

# Step 8: Keep only needed columns
df = df[['sigheight', 'windspeed']]

# Step 9: Rename
df.columns = ['wave_height', 'wind_speed']

# Step 10: Normalize
df['wave_height'] = (df['wave_height'] - df['wave_height'].min()) / (df['wave_height'].max() - df['wave_height'].min())
df['wind_speed'] = (df['wind_speed'] - df['wind_speed'].min()) / (df['wind_speed'].max() - df['wind_speed'].min())

# Step 11: Save processed file
df.to_csv("data/processed_sensor.csv", index=False)

print("✅ Sensor data processed successfully")