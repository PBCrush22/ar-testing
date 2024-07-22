import subprocess
import pandas as pd
import numpy as np
import time

def scan_wifi():
    result = subprocess.run(['netsh', 'wlan', 'show', 'network', 'mode=Bssid'], stdout=subprocess.PIPE)
    output = result.stdout.decode('latin-1')
    
    data = []
    lines = output.split('\n')
    current_ssid = None
    current_mac = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("SSID"):
            parts = line.split(': ')
            if len(parts) > 1:
                current_ssid = parts[1].strip()
                if not current_ssid.startswith("eduroam"):
                    current_ssid = None
        elif line.startswith("BSSID"):
            parts = line.split(': ')
            if len(parts) > 1:
                current_mac = parts[1].strip()
        elif line.startswith("Signal"):
            parts = line.split(': ')
            if len(parts) > 1:
                signal = int(parts[1].replace('%', '').strip())
                rssi = -100 + signal / 2
                if current_ssid and current_mac:
                    data.append((current_ssid, current_mac, rssi))
    return data

def filter_outliers(df):
    Q1 = df['signal'].quantile(0.25)
    Q3 = df['signal'].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[(df['signal'] >= Q1 - 1.5 * IQR) & (df['signal'] <= Q3 + 1.5 * IQR)]
    return filtered_df

def collect_real_time_data(samples=10):
    all_data = []
    for _ in range(samples):
        wifi_data = scan_wifi()
        all_data.extend(wifi_data)
        time.sleep(1)
    
    df = pd.DataFrame(all_data, columns=['ssid', 'mac', 'signal'])
    df = filter_outliers(df)
    averaged_df = df.groupby(['ssid', 'mac'], as_index=False).mean()
    return averaged_df

# Load the fingerprint data
df = pd.read_csv('wifi_fingerprint_data_all_locations.csv')

# Pivot the data
df_pivot = df.pivot_table(index='location', columns='mac', values='signal', fill_value=-100)


# Save pivot table to CSV
df_pivot.to_csv('pivot_table_output.csv', index=True)

print("Pivot table saved to 'pivot_table_output.csv'.")


def weighted_distance(x, y):
    weights = np.exp(np.abs(x) / 20)  # More weight to stronger signals
    return np.sqrt(np.sum(weights * (x - y) ** 2))

def predict_location():
    # Collect real-time WiFi data
    real_time_df = collect_real_time_data(samples=10)
    
    # Print the collected real-time WiFi data
    print("Real-time WiFi data collected:")
    for ssid, mac, rssi in real_time_df.values:
        print(f"SSID: {ssid}, MAC: {mac}, RSSI: {rssi}")

    # Pivot the real-time data to get a single row
    real_time_pivot = real_time_df.pivot_table(index=lambda x: 0, columns='mac', values='signal', fill_value=-100)
    
    # Ensure the real-time pivot table has the same columns as the training data
    real_time_pivot = real_time_pivot.reindex(columns=df_pivot.columns, fill_value=-100)
    
    real_time_pivot.to_csv('real_time_pivot_table.csv', index=True)
    # Print the real_time_pivot DataFrame
    print("\nReal-time pivot table:")
    print(real_time_pivot)
    
    # Convert the real-time pivot table to a single row format
    X_test = real_time_pivot.values.reshape(1, -1)
    
    # Compare each row of df_pivot with X_test and find the most similar one
    min_distance = float('inf')
    predicted_location = None
    for location in df_pivot.index:
        distance = weighted_distance(df_pivot.loc[location].values, X_test[0])
        if distance < min_distance:
            min_distance = distance
            predicted_location = location

    # Print the training data for the predicted location
    predicted_location_data = df[df['location'] == predicted_location]
    print(f"\nTraining data for predicted location ({predicted_location}):")
    for _, row in predicted_location_data.iterrows():
        print(f"SSID: {row['ssid']}, MAC: {row['mac']}, RSSI: {row['signal']}")
    
    return predicted_location

# Predict the location based on real-time WiFi data
predicted_location = predict_location()
print(f"\nPredicted location: {predicted_location}")