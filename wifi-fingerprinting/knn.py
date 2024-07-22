import subprocess
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

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

def normalize_signals(df):
    df_normalized = df.copy()
    df_normalized['signal'] = (df['signal'] + 100) / 100  # Normalize to a range between 0 and 1
    return df_normalized

# Load the fingerprint data
df = pd.read_csv('wifi_fingerprint_data_all_locations.csv')

# Normalize the fingerprint data
df_normalized = normalize_signals(df)

# Pivot the data
df_pivot = df_normalized.pivot_table(index='location', columns='mac', values='signal', fill_value=-1)

print("\ndf pivot table:")
print(df_pivot)

def predict_location():
    # Collect real-time WiFi data
    real_time_df = collect_real_time_data(samples=10)
    
    # Normalize the real-time WiFi data
    real_time_normalized = normalize_signals(real_time_df)
    
    # Print the collected real-time WiFi data
    print("Real-time WiFi data collected:")
    for ssid, mac, rssi in real_time_df.values:
        print(f"SSID: {ssid}, MAC: {mac}, RSSI: {rssi}")

    # Pivot the real-time data to get a single row
    real_time_pivot = real_time_normalized.pivot_table(index=lambda x: 0, columns='mac', values='signal', fill_value=-1)
    
    # Ensure the real-time pivot table has the same columns as the training data
    real_time_pivot = real_time_pivot.reindex(columns=df_pivot.columns, fill_value=-1)
    
    # Print the real_time_pivot DataFrame
    print("\nReal-time pivot table:")
    print(real_time_pivot)
    
    # Convert the real-time pivot table to a single row format
    X_test = real_time_pivot.values.reshape(1, -1)
    
    # Use k-NN to predict the location
    knn = NearestNeighbors(n_neighbors=3, metric='manhattan')
    knn.fit(df_pivot)
    
    distances, indices = knn.kneighbors(X_test)
    nearest_locations = df_pivot.index[indices.flatten()]
    
    predicted_location = nearest_locations.value_counts().idxmax()
    
    # Print the training data for the predicted location
    predicted_location_data = df[df['location'] == predicted_location]
    print(f"\nTraining data for predicted location ({predicted_location}):")
    for _, row in predicted_location_data.iterrows():
        print(f"SSID: {row['ssid']}, MAC: {row['mac']}, RSSI: {row['signal']}")
    
    return predicted_location

# Predict the location based on real-time WiFi data
predicted_location = predict_location()
print(f"\nPredicted location: {predicted_location}")
