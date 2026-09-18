import os
import zipfile
import subprocess
import shutil

def download_cmapss():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'cmapss')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading CMAPSS dataset to {data_dir}...")
    
    # We will use the 'palbha/cmapss-jet-engine-simulated-data' dataset which contains the standard text files
    dataset_name = "palbha/cmapss-jet-engine-simulated-data"
    
    try:
        # Run kaggle cli command
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", data_dir],
            check=True
        )
        print("Download successful. Extracting...")
        
        # Unzip the downloaded file
        zip_path = os.path.join(data_dir, f"{dataset_name.split('/')[1]}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path) # clean up zip
            print(f"Dataset extracted successfully to {data_dir}.")
        else:
            print("Warning: Zip file not found after download.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset using Kaggle CLI. Make sure Kaggle API is configured.")
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_cmapss()
