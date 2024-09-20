"""
Inputs to the Function:

	•	total_sequences: (int) The total number of sequences you want to generate.
	•	batch_size: (int) The batch size for sequence generation.
	•	n_sequences: (int) The number of sequences in MSA to subsample.
	•	output_csv_file: (str) The path to the output CSV file where the generated sequences will be saved.
    •	to_device: (str) Device to run the model, cuda or cpu.

Outputs of the Function:

	•	CSV File: A CSV file containing the generated sequences with their IDs and lengths.
"""

import random
import csv
import argparse
from evodiff.pretrained import MSA_OA_DM_MAXSUB
from evodiff.generate_msa import generate_msa


def generate_unconditional_msa_sequences(total_sequences, batch_size, n_sequences, output_csv_file, to_device='cuda'):
    """
    Generates AMP sequences using EvoDiff Unconditional Generation with MSA_OA_DM_MAXSUB and saves them to a CSV file.

    Args:
        total_sequences (int): Total number of sequences to generate.
        batch_size (int): Batch size for sequence generation.
        n_sequences (int): Number of sequences to generate in each batch.
        output_csv_file (str): Path to the output CSV file.
        to_device: (str) Device to run the model, cuda or cpu.
    """
    checkpoint = MSA_OA_DM_MAXSUB()
    model, collater, tokenizer, scheme = checkpoint
    model.to(to_device)
    
    sequences_data = []
    sequence_id_counter = 0

    for _ in range(total_sequences):
        seq_len = random.randint(15, 35)  # Randomly choose a length between 15 and 35
        _, untokenized = generate_msa(model, tokenizer, batch_size=batch_size, n_sequences=n_sequences, seq_length=seq_len, penalty_value=1, device=to_device,
                                      start_query=False, start_msa=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/')
        for generated_sequence in untokenized:
            seq_tem = str(generated_sequence)
            seq_tem = seq_tem[3:seq_len + 3]
            seq_tem = seq_tem[:-1]

            # Store each sequence's data in a dictionary
            sequence_data = {
                "ID": sequence_id_counter,
                "Sequence": seq_tem
            }
            sequences_data.append(sequence_data)
            sequence_id_counter += 1
            

            # Break if total sequences reached
            if sequence_id_counter >= total_sequences:
                break
        # Break the outer loop as well when total sequences reached
        if sequence_id_counter >= total_sequences:
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
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for sequence generation")
    parser.add_argument('--n_sequences', type=int, default=64, help="The number of sequences in MSA to subsample")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--to_device', type=str, default="cuda", help="Device to run the model, cuda or cpu")
    args = parser.parse_args()
    generate_unconditional_msa_sequences(args.total_sequences, args.batch_size, args.n_sequences, args.output_csv_file, args.to_device)




