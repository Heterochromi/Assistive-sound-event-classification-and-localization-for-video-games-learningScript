import os
import pandas as pd
import shutil

def combine_datasets():
    """
    Combines multiple datasets of spectrogram images and their corresponding CSV files
    into a single dataset. It copies images to a new directory, prefixing them
    to avoid name collisions, and creates a new combined CSV file with updated
    filenames.
    """
    # Base path to keep the code clean
    base_path = '/home/baraa/Desktop/testroch/DSED_strong_label/dataset'

    # Configuration for the different datasets to be combined
    datasets = [
        {
            "prefix": "500ms",
            "mel_dir": os.path.join(base_path, "audio/eval/500ms_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/500ms.tsv_weak_labels.csv")
        },
        {
            "prefix": "5500ms",
            "mel_dir": os.path.join(base_path, "audio/eval/5500ms_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/5500ms.tsv_weak_labels.csv")
        },
        {
            "prefix": "9500ms",
            "mel_dir": os.path.join(base_path, "audio/eval/9500ms_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/9500ms.tsv_weak_labels.csv")
        }, 
        {
            "prefix": "distorted_clipping",
            "mel_dir": os.path.join(base_path, "audio/eval/distorted_clipping_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/distorted_clipping.tsv_weak_labels.csv")
        },
        {
            "prefix": "distorted_lowpass_filter",
            "mel_dir": os.path.join(base_path, "audio/eval/distorted_lowpass_filter_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/distorted_lowpass_filter.tsv_weak_labels.csv")
        },
        {
            "prefix": "distorted_highpass_filter_mel",
            "mel_dir": os.path.join(base_path, "audio/eval/distorted_highpass_filter_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/distorted_highpass_filter.tsv_weak_labels.csv")
        },
         {
            "prefix": "distorted_drc",
            "mel_dir": os.path.join(base_path, "audio/eval/distorted_drc_mel"),
            "csv_path": os.path.join(base_path, "metadata/eval/distorted_drc.tsv_weak_labels.csv")
        }
    ]

    # Define destination paths for the combined data
    combined_mel_dir = os.path.join(base_path, "audio/eval/combined_mel")
    combined_csv_path = os.path.join(base_path, "metadata/eval/combined_weak_labels.csv")

    # Create the destination directory for mel spectrograms; exist_ok=True prevents an error if it already exists
    os.makedirs(combined_mel_dir, exist_ok=True)

    # List to hold the dataframes for each dataset
    all_dfs = []
    
    print("Starting the dataset combination process...")

    # Process each dataset
    for dataset_info in datasets:
        prefix = dataset_info["prefix"]
        mel_dir = dataset_info["mel_dir"]
        csv_path = dataset_info["csv_path"]

        print(f"Processing dataset with prefix: {prefix}")

        # Check for the existence of source directories and files to prevent errors
        if not os.path.isdir(mel_dir):
            print(f"Warning: Mel directory not found at {mel_dir}. Skipping this dataset.")
            continue
        if not os.path.isfile(csv_path):
            print(f"Warning: CSV file not found at {csv_path}. Skipping this dataset.")
            continue

        # Read the dataset's CSV file
        df = pd.read_csv(csv_path)
        
        # Keep track of new filenames
        new_filenames = []

        # Iterate over each row in the dataframe to process each file
        for _, row in df.iterrows():
            original_filename = row['filename']
            # Create a new filename with the prefix to ensure it's unique
            original_filename = original_filename.replace('.jams', '.png')  # Assuming the original filenames are .jams files
            new_filename = f"{prefix}_{original_filename}"
            new_filenames.append(new_filename)


            # Define the source and destination paths for the image file
            source_path = os.path.join(mel_dir, original_filename)
            destination_path = os.path.join(combined_mel_dir, new_filename)

            # Copy the file if it exists
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
            else:
                print(f"Warning: File {original_filename} not found in {mel_dir}")
        
        # Update the 'filename' column with the new prefixed filenames
        df['filename'] = new_filenames
        all_dfs.append(df)

    # Combine all dataframes into one
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save the combined dataframe to a new CSV file
        combined_df.to_csv(combined_csv_path, index=False)
        
        print("\nDataset combination complete!")
        print(f"Combined mel spectrograms are in: {combined_mel_dir}")
        print(f"Combined CSV is saved at: {combined_csv_path}")
        print(f"Total samples in combined dataset: {len(combined_df)}")
    else:
        print("No datasets were processed. Please check the paths.")

if __name__ == '__main__':
    combine_datasets()




