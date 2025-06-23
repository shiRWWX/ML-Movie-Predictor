import zipfile
import os

# List of zip file paths
zip_files = [r'C:\Users\shiva\OneDrive\Desktop\MinorProject\tmdb_5000_movies.csv.zip', r'C:\Users\shiva\OneDrive\Desktop\MinorProject\tmdb_5000_credits.csv.zip']

# Destination folder for extracted CSVs
extract_to_folder = 'dataset'
os.makedirs(extract_to_folder, exist_ok=True)

# Loop over zip files
for zip_file_path in zip_files:
    if not os.path.exists(zip_file_path):
        print(f"❌ File not found: {zip_file_path}")
        continue

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.csv'):
                zip_ref.extract(file, extract_to_folder)
                print(f"✅ Extracted {file} from {zip_file_path} to {extract_to_folder}/")

print("All CSV files extracted successfully!")