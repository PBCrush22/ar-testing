import subprocess
import pandas as pd
import time
import os

def scan_wifi():
    # Using `netsh` command to scan WiFi (Windows example)
    result = subprocess.run(['netsh', 'wlan', 'show', 'network', 'mode=Bssid'], stdout=subprocess.PIPE)
    output = result.stdout.decode('latin-1')
    
    # Parse the output to extract SSID, BSSID, and RSSI
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
                # Uncomment the following lines if you want to filter SSIDs
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
                # Convert signal strength to RSSI
                rssi = -100 + signal / 2
                if current_ssid and current_mac:
                    data.append((current_ssid, current_mac, rssi))
                    print(f"Captured: SSID={current_ssid}, MAC={current_mac}, RSSI={rssi}")  # Debug statement
    
    return data

def collect_data(location, samples=1):
    all_data = []
    for _ in range(samples):
        wifi_data = scan_wifi()
        all_data.extend(wifi_data)
        time.sleep(1)  # Pause between scans
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['ssid', 'mac', 'signal'])
    # Group by SSID and MAC address, and calculate the mean RSSI
    averaged_df = df.groupby(['ssid', 'mac'], as_index=False).mean()
    # Add the location information
    averaged_df['location'] = location
    return averaged_df

# List of locations (example range)
# locations = [x for x in range(1, 17, 3)]
locations = [10]
# Initialize an empty list to store data frames
all_data_frames = []

# Loop through each location
for location in locations:
    input(f"Move to {location} and press Enter to start collecting data...")
    
    # Collect data for the current location
    location_data = collect_data(location)
    
    # Append the data frame to the list
    all_data_frames.append(location_data)

# Concatenate all data frames into one
all_data = pd.concat(all_data_frames, ignore_index=True)

all_data = all_data[all_data['location'] != 0]

# Check if the file exists
if os.path.isfile('wifi_fingerprint_data_all_locations.csv'):
    # Load the existing data from the CSV file
    existing_data = pd.read_csv('wifi_fingerprint_data_all_locations.csv')

    # Append the new data to the existing data
    combined_data = pd.concat([existing_data, all_data], ignore_index=True)
else:
    combined_data = all_data

# Save the combined data to the CSV file
combined_data.to_csv('wifi_fingerprint_data_all_locations.csv', index=False)

print("Data collection complete. The data has been saved to 'wifi_fingerprint_data_all_locations.csv'.")