"""
	1.	Function Definition: The core logic is wrapped in a function generate_unconditional_sequences that takes total_sequences, batch_size, and output_file as arguments.
	2.	Command-Line Interface: The script uses argparse to handle command-line arguments, making it easy to run from the command line.
	3.	File Writing: The function saves the generated sequences to a CSV file specified by the user.
"""

import random
import csv
import argparse
from evodiff.generate import generate_oaardm
from evodiff.pretrained import OA_DM_640M

def generate_unconditional_sequences(total_sequences, batch_size, output_file, to_device='cuda'):
    checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model.to(to_device)
    
    # iterations = total_sequences // batch_size + (total_sequences % batch_size > 0)
    sequences_data = []
    sequence_id_counter = 0

    for _ in range(total_sequences):
        seq_len = random.randint(15, 35)  # Randomly choose a length between 15 and 35
        _, generated_sequences = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device=to_device)

        for generated_sequence in generated_sequences:
            # Store each sequence's data in a dictionary
            sequence_data = {
                "ID": sequence_id_counter,
                "Sequence": generated_sequence
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
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sequences_data[0].keys())
        writer.writeheader()
        writer.writerows(sequences_data)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP sequences using EvoDiff Unconditional Generation")
    parser.add_argument('--total_sequences', type=int, required=True, help="Total number of sequences to generate")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for sequence generation")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--to_device', type=str, default="cuda", help="Device to run the model, cuda or cpu")
    
    args = parser.parse_args()
    generate_unconditional_sequences(args.total_sequences, args.batch_size, args.output_file, args.to_device)
    



    
