import os
import shutil
import pandas as pd

# Define paths
input_dir = '/content/drive/MyDrive/AudioBad/all'  # Directory where the audio files are currently located
output_dir = '/content/drive/MyDrive/AudioBad/org'  # Directory where you want to organize the files

# Read the CSV file
csv_file = '/content/drive/MyDrive/AudioBad/merged_file.csv'
df = pd.read_csv(csv_file)

# Iterate through each row in the CSV file and organize the files
for index, row in df.iterrows():
    audio_file = row['file-name']  # Assuming 'audio_file' is the column with the audio file names
    bird_species = row['sp']  # Assuming 'bird_species' is the column with bird species labels

    # Construct source and destination paths
    source_path = os.path.join(input_dir, audio_file)
    destination_path = os.path.join(output_dir, bird_species)

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # Move the audio file to the destination folder
    shutil.move(source_path, os.path.join(destination_path, os.path.basename(audio_file)))

print("Audio files organized based on bird species.")
