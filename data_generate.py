import os
import shutil

# Path to the extracted RAVDESS folder
source_dir = 'Audio_Speech_Actors_01-24'
target_dir = 'data'

os.makedirs(target_dir, exist_ok=True)

for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                full_path = os.path.join(folder_path, file)
                shutil.copy(full_path, target_dir)

print(f"Copied all WAV files to {target_dir}")