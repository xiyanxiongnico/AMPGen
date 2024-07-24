

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:24:54 2024

Inputs to the Function:

	•	directory_path: (str) The path to the directory containing MSA files.
	•	output_excel_file: (str) The path to the output Excel file where the generated sequences will be saved.
	•	max_retries: (int, optional) The maximum number of retries for each file. Default is 5.

Outputs of the Function:

	•	Excel File: An Excel file containing the results with columns for filename, number of sequences, reference sequence, reference length, generated sequence, and generated length.

Command-Line Usage Example
generate_conditional_msa_sequences --directory_path path/to/directory --output_csv_file path/to/output_file.csv --max_retries 5
@author: xiyanxiong

"""
# src/generation/conditional_generation_msa.py

import os
import random
import re
import pandas as pd
from evodiff.pretrained import MSA_OA_DM_MAXSUB
from evodiff.generate_msa import generate_query_oadm_msa_simple

def generate_conditional_msa_sequences(directory_path, output_csv_file, max_retries=5):
    """
    Generates AMP sequences using EvoDiff Conditional Generation with MSA_OA_DM_MAXSUB and saves the results to a CSV file.

    Args:
        directory_path (str): Path to the directory containing MSA files.
        output_csv_file (str): Path to the output CSV file.
        max_retries (int): Maximum number of retries for each file (default: 5).
    """
    checkpoint = MSA_OA_DM_MAXSUB()
    model, collater, tokenizer, scheme = checkpoint
    model.to('cuda')
    
    output_data = []

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".a3m"):
            path_to_msa = os.path.join(directory_path, filename)
            
            # Read the file to determine the number of sequences
            with open(path_to_msa, 'r') as file:
                sequences = [line.strip() for line in file if line.startswith(">")]
                n_sequences = len(sequences) - 1
                if n_sequences > 64:
                    n_sequences = 64
            
            # Read the first sequence (reference sequence)
            with open(path_to_msa, 'r') as file:
                msa_content = file.read()
                reference_sequence = re.search(r"^>[^\n]*\n([^>]*)", msa_content, re.MULTILINE).group(1).replace('\n', '')
                reference_length = len(reference_sequence.replace('-', ''))  # Remove gaps and get length

            # Run the model 5 times with random seq_length each time
            for _ in range(5):
                retries = 0
                while retries < max_retries:
                    try:
                        seq_length = random.randint(15, 35)
                        selection_type = 'MaxHamming'  # or 'random'; MSA subsampling scheme
                        
                        # Running the model
                        tokenized_sample, generated_sequence = generate_query_oadm_msa_simple(
                            path_to_msa, model, tokenizer, n_sequences, seq_length, device='cuda', selection_type=selection_type)
                        
                        clean_generated_sequence = re.sub('[!-]', '', generated_sequence[0][0])
                        generated_length = len(clean_generated_sequence)

                        output_data.append([filename, n_sequences, reference_sequence, reference_length, clean_generated_sequence, generated_length])
                        break  # Break out of the retry loop if successful
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        retries += 1
                        if retries == max_retries:
                            print(f"Failed to process {filename} after {max_retries} retries.")
                        else:
                            print(f"Retrying {filename} ({retries}/{max_retries})...")

    # Creating a CSV file with the results
    df = pd.DataFrame(output_data, columns=["Filename", "Number of Sequences", "Reference Sequence", "Reference Length", "Generated Sequence", "Generated Length"])
    df.to_csv(output_csv_file, index=False)

    print(f"Data saved to {output_csv_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate AMP sequences using EvoDiff Conditional Generation with MSA_OA_DM_MAXSUB")
    parser.add_argument('--directory_path', type=str, required=True, help="Path to the directory containing MSA files")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--max_retries', type=int, default=5, help="Maximum number of retries for each file")
    
    args = parser.parse_args()
    generate_conditional_msa_sequences(args.directory_path, args.output_csv_file, args.max_retries)
    

