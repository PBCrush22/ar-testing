import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Function to scan WiFi networks and retrieve signal strengths
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

# Function to filter outliers based on signal strength
def filter_outliers(df):
    Q1 = df['signal'].quantile(0.25)
    Q3 = df['signal'].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[(df['signal'] >= Q1 - 1.5 * IQR) & (df['signal'] <= Q3 + 1.5 * IQR)]
    return filtered_df

# Function to collect and average real-time WiFi data
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

# Load the fingerprinted WiFi data
df = pd.read_csv('wifi_fingerprint_data_all_locations.csv')

# Pivot the data to create a fingerprint (pivot) table
df_pivot = df.pivot_table(index='location', columns='mac', values='signal', fill_value=-100)

# Prepare data for Random Forest
X = df_pivot.values  # Features (signal strengths)
y = df_pivot.index    # Labels (locations)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Function to predict the location based on real-time WiFi data using Random Forest
def predict_location_rf(clf, real_time_df):
    # Pivot the real-time data to get a single row (similar to fingerprint format)
    real_time_pivot = real_time_df.pivot_table(index=lambda x: 0, columns='mac', values='signal', fill_value=-100)
    
    # Ensure the real-time pivot table has the same columns as the training data
    real_time_pivot = real_time_pivot.reindex(columns=df_pivot.columns, fill_value=-100)
    
    # Convert the real-time pivot table to a single row format
    X_test = real_time_pivot.values.reshape(1, -1)
    
    # Predict the location
    predicted_location = clf.predict(X_test)[0]
    
    return predicted_location

# Function to print real-time WiFi data for debugging
def print_real_time_data(real_time_df):
    print("Real-time WiFi data collected:")
    for ssid, mac, rssi in real_time_df.values:
        print(f"SSID: {ssid}, MAC: {mac}, RSSI: {rssi}")

# Predict the location based on real-time WiFi data using Random Forest
def predict_location():
    # Collect real-time WiFi data
    real_time_df = collect_real_time_data(samples=10)
    
    # Print real-time data for debugging
    print_real_time_data(real_time_df)
    
    # Predict location using Random Forest
    predicted_location = predict_location_rf(clf, real_time_df)
    
    # Print the predicted location
    print(f"\nPredicted location: {predicted_location}")
    
    return predicted_location

# Predict the location
predicted_location = predict_location()
