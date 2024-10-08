import pandas as pd
from tqdm import tqdm
import os

file_path = 'data/OII_US_POST_BODY.csv'

# Get the total file size in bytes
file_size = os.path.getsize(file_path)

# Define the chunk size (number of rows per chunk)
chunk_size = 10000

# Initialize an empty list to store chunks
data = []

# Track the number of bytes read
bytes_read = 0

# Use tqdm to display the progress based on file size
with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading CSV by file size") as pbar:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        data.append(chunk)
        # Estimate bytes read so far based on chunk size
        bytes_read += chunk.memory_usage(deep=True).sum()
        # Update the progress bar with the number of bytes read
        pbar.update(bytes_read - pbar.n)  # Ensure only new bytes are updated

# Combine all chunks into a single DataFrame
print("Combining chunks into a single DataFrame...")
body = pd.concat(data, ignore_index=True)

salary_sample = pd.read_parquet('data/salary_sample_20k.parquet.gzip')

print("Merging the salary_sample and body DataFrames...")
salary_sample_body = salary_sample.merge(body, left_on='ID', right_on='ID', how='left')

print("Saving salary_sample_body to parquet...")
salary_sample_body.to_parquet('data/salary_sample_body.parquet.gzip', compression='gzip')

print("Loading the no salary sample...")
all_sample = pd.read_parquet('data/nosalary_sample_20k.parquet.gzip')

print("Merging the no salary sample and body DataFrames...")
all_sample_body = all_sample.merge(body, left_on='ID', right_on='ID', how='left')

print("Saving no salary sample body to parquet...")
all_sample_body.to_parquet('data/nosalary_sample_body.parquet.gzip', compression='gzip')

print("Saving body to parquet...")
body.to_parquet('data/body.parquet.gzip', compression='gzip')