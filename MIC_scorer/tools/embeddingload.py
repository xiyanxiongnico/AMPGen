import os
import torch
import pandas as pd

# Define the folder path
folder_path = '/media/zzh/data/AMP/train_data/regression/5_65_stpa/3B'
csv_file_path = '/media/zzh/data/AMP/train_data/regression/ave_5-65regression_stpa_all.csv'  # The file path of the CSV containing the Sequence and logMIC columns

# Get all .pt file names and sort them in numerical order
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')], key=lambda x: int(x.split('.')[0]))

# Initialize an empty list to store mean_representations
mean_representations_list = []

# Sequentially load each .pt file and extract mean_representations
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    mean_representation = torch.load(file_path)['mean_representations'][36].numpy()
    mean_representation = mean_representation.tolist()  # Convert to a list
    mean_representations_list.append(mean_representation)

# create a DataFrame
mean_representations_df = pd.DataFrame(mean_representations_list)

# Load the CSV file containing the Sequence and logMIC columns
additional_data_df = pd.read_csv(csv_file_path)

# Ensure the number of files matches the number of rows
if len(mean_representations_df) != len(additional_data_df):
    raise ValueError("The number of rows in the additional data CSV does not match the number of .pt files")

# Insert the Sequence column as the first column
final_df = pd.concat([additional_data_df['Sequence'], mean_representations_df], axis=1)

# Insert the logMIC column as the last column
final_df['logMIC'] = additional_data_df['logMIC']

# Save the final DataFrame as a CSV file
final_df.to_csv('/media/zzh/data/AMP/train_data/regression/5_65_ecoli/5_65_stpa_mean_representations.csv', index=False)

print("Final DataFrame has been saved to 'final_mean_representations.csv'")
