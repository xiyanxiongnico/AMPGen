# Generate sequences with random length ranging from 15-35 
"""
Created on Mon May 13 21:24:54 2024

	1.	Function Definition: The core logic is wrapped in a function generate_unconditional_sequences that takes total_sequences, batch_size, and output_file as arguments.
	2.	Command-Line Interface: The script uses argparse to handle command-line arguments, making it easy to run from the command line.
	3.	File Writing: The function saves the generated sequences to a CSV file specified by the user.

    command line usage example: 
    unconditional_generation --total_sequences 100 --batch_size 10 --output_file /path/to/output.csv
@author: xiyanxiong
"""


# src/generation/unconditional_generation.py

import random
import csv
import argparse
from evodiff.generate import generate_oaardm
from evodiff.pretrained import OA_DM_640M

def generate_unconditional_sequences(total_sequences, batch_size, output_file):
    checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model.to('cuda')
    
    iterations = total_sequences // batch_size + (total_sequences % batch_size > 0)
    sequences_data = []
    sequence_id_counter = 1

    for _ in range(iterations):
        seq_len = random.randint(15, 35)  # Randomly choose a length between 15 and 35
        _, generated_sequences = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device='cuda')

        for generated_sequence in generated_sequences:
            # Store each sequence's data in a dictionary
            sequence_data = {
                "sequence_id": f"seq_{sequence_id_counter}",
                "Generated Sequence": generated_sequence,
                "length": len(generated_sequence),
                "MIC": "",  # Placeholder for future data
                "hydrophobicity": "",  # Placeholder for future data
                "net_charge": ""  # Placeholder for future data
            }
            
            sequences_data.append(sequence_data)
            sequence_id_counter += 1

            # Break if total sequences reached
            if sequence_id_counter > total_sequences:
                break

    # Write the data to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sequences_data[0].keys())
        writer.writeheader()
        writer.writerows(sequences_data)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP sequences using EvoDiff Unconditional Generation")
    parser.add_argument('--total_sequences', type=int, required=True, help="Total number of sequences to generate")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for sequence generation")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file")
    
    args = parser.parse_args()
    generate_unconditional_sequences(args.total_sequences, args.batch_size, args.output_file)
    



    
