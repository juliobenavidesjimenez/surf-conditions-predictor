"""
Data collection module for Zarautz surf conditions
Using Puertos del Estado API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# Constants for the API endpoints and parameters
# Ubication ID for Zarautz
# Surf spots coordinates
SURF_SPOTS = {
    "zarautz": {"lat": 43.28, "lon": -2.17},
    "mundaka": {"lat": 43.41, "lon": -2.69},
    "sopelana": {"lat": 43.38, "lon": -3.00}
}

SURF_VARIABLES = [
    "wave_height",
    "wave_direction", 
    "wave_period",
    "wind_wave_height",
    "wind_wave_direction",
    "wind_wave_period"
]

def fetch_surf_data(spot_name, start_date, end_date):
    """
    Fetch surf data from Puertos del Estado API for Zarautz location
    between start_date and end_date.
    
    Parameters:
        spot_name: str, name of surf spot (e.g., 'zarautz', 'mundaka')
        start_date: str, start date 'YYYY-MM-DD'
        end_date: str, end date 'YYYY-MM-DD'
    
    Returns:
        dict: JSON data from API
    """

    # Build API URL
    base_url = "https://marine-api.open-meteo.com/v1/marine"

    # Parameters for the API request
    params = {
            "latitude": SURF_SPOTS[spot_name]["lat"],
            "longitude": SURF_SPOTS[spot_name]["lon"],
            "hourly": ",".join(SURF_VARIABLES),
            "start_date": start_date,
            "end_date": end_date
        }

    # Make the API request
    response = requests.get(base_url, params=params)

    # Check for successful response
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        return None

def json_to_dataframe(json_data):
    """
    Convert JSON data from API to pandas DataFrame.
    
    Parameters:
        json_data: JSON data from API
    
    Returns: A pandas DataFrame with surf data
    """
    if not json_data or 'hourly' not in json_data:
        print("Invalid JSAON Data")
        return None
    return pd.DataFrame(json_data['hourly'])

def save_dataframe_to_csv(df, filename):
    """
    Save DataFrame to CSV File

    Parameters:
        df: pandas DataFrame
        filename: str, name for the CSV file
    """
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    print(f"DataFrame saved to {filepath}")

if __name__ == "__main__":
    print("Testing data collection...")

    spot = "zarautz"
    start = "2024-01-01"
    end = "2024-12-05"
    
    # Fetch data
    data = fetch_surf_data(spot, start, end)
    
    if data:
        print("Data fetched successfully!")

        # Convert to DataFrame
        df = json_to_dataframe(data)

        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst 5 rows:\n{df.head()}")

        # Save to CSV
        filename = f"{spot}_{start}_{end}.csv"
        save_dataframe_to_csv(df, filename)

    else:
        print("Failed to fetch data")