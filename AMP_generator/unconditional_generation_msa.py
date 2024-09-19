# src/generation/unconditional_generation_msa.py
"""
Inputs to the Function:

	•	total_sequences: (int) The total number of sequences you want to generate.
	•	batch_size: (int) The batch size for sequence generation.
	•	n_sequences: (int) The number of sequences in MSA to subsample.
	•	output_csv_file: (str) The path to the output CSV file where the generated sequences will be saved.

Outputs of the Function:

	•	CSV File: A CSV file containing the generated sequences with their IDs and lengths.

Command-Line Usage Example
unconditional_generation_msa --total_sequences 100 --batch_size 10 --n_sequences 64 --output_csv_file /path/to/output.csv
"""

import random
import csv
import argparse
from evodiff.pretrained import MSA_OA_DM_MAXSUB
from evodiff.generate_msa import generate_msa


def generate_unconditional_msa_sequences(total_sequences, batch_size, n_sequences, output_csv_file):
    """
    Generates AMP sequences using EvoDiff Unconditional Generation with MSA_OA_DM_MAXSUB and saves them to a CSV file.

    Args:
        total_sequences (int): Total number of sequences to generate.
        batch_size (int): Batch size for sequence generation.
        n_sequences (int): Number of sequences to generate in each batch.
        output_csv_file (str): Path to the output CSV file.
    """
    checkpoint = MSA_OA_DM_MAXSUB()
    model, collater, tokenizer, scheme = checkpoint
    model.to('cuda')
    
    sequences_data = []
    sequence_id_counter = 1

    for _ in range(total_sequences):
        seq_len = random.randint(15, 35)  # Randomly choose a length between 15 and 35
        _, untokenized = generate_msa(model, tokenizer, batch_size=batch_size, n_sequences=n_sequences, seq_length=seq_len, penalty_value=2, device='cuda',
                                      start_query=False, start_msa=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/')
        for generated_sequence in untokenized:
            seq_tem = str(generated_sequence)
            seq_tem = seq_tem[3:seq_len + 3]

            # Store each sequence's data in a dictionary
            sequence_data = {
                "sequence_id": f"seq_{sequence_id_counter}",
                "Generated Sequence": seq_tem,
                "seq_len": seq_len,
            }
            sequences_data.append(sequence_data)
            sequence_id_counter += 1

            # Break if total sequences reached
            if sequence_id_counter > total_sequences:
                break

    # Write the data to a CSV file
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sequences_data[0].keys())
        writer.writeheader()
        writer.writerows(sequences_data)

    print(f"Data saved to {output_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP sequences using EvoDiff Unconditional Generation with MSA_OA_DM_MAXSUB")
    parser.add_argument('--total_sequences', type=int, required=True, help="Total number of sequences to generate")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for sequence generation")
    parser.add_argument('--n_sequences', type=int, required=True, help="Number of sequences to generate in each batch")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to the output CSV file")
    
    args = parser.parse_args()
    generate_unconditional_msa_sequences(args.total_sequences, args.batch_size, args.n_sequences, args.output_csv_file)




