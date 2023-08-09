from bs4 import BeautifulSoup
import pandas as pd
import requests
from pprint import pprint

url_species = 'https://xeno-canto.org/api/2/recordings?query=troglodytes+troglodytes'
response = requests.get(url_species)

# Parse the JSON response
data = response.json()

# Extract the "recordings" list from the data
recordings_list = data['recordings']

# Create a DataFrame from the "recordings" list
df = pd.DataFrame(recordings_list)

# Write the DataFrame to a CSV file
df.to_csv('bird_dataset2.csv', index=False)

# Print the first few rows of the DataFrame (optional)
pprint(df.head())

def download(url, file_name):
    # Send a GET request to download the audio file
    response = requests.get(url)
    file_name = f"{file_name}"

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the audio content to a local file
        with open(file_name, "wb") as f:
            f.write(response.content)
        print("Audio downloaded successfully.")
    else:
        print(f"Failed to download audio. Status code: {response.status_code}")

for file, file_name in zip(df["file"], df["file-name"]):
    download(file, file_name)
    print("Downloaded {file_name}".format(file_name=file_name))