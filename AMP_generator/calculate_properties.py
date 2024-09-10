
# src/analysis/calculate_properties.py

"""

Command-Line Usage:
    calculate_properties --input_csv_file /path/to/input.csv --output_csv_file /path/to/output.csv

"""


from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq
import pandas as pd
import argparse

class CustomPA:
    def __init__(self, peptide_sequence: str):
        self.sequence = peptide_sequence
        self.peptide = Seq(peptide_sequence)
        self.analysis = ProteinAnalysis(peptide_sequence)

    def calculate_molecular_weight(self) -> float:
        """
        Calculate the molecular weight of the peptide.
        :return: Molecular weight
        """
        return molecular_weight(self.peptide, seq_type='protein')

    def calculate_net_charge(self, pH: float = 7.0) -> float:
        """
        Calculate the net charge of the peptide at a given pH using pKa values from the CRC Handbook of Chemistry and Physics, 87th edition.
        :param pH: pH value, default is 7.0
        :return: Net charge
        """
        pKa_values = {
            'C-terminus': 2.34,
            'N-terminus': 9.69,
            'D': 3.86,
            'E': 4.25,
            'C': 8.33,
            'Y': 10.07,
            'H': 6.00,
            'K': 10.54,
            'R': 12.48
        }

        net_charge = 0.0

        # Calculate the charge of N-terminus and C-terminus
        net_charge += 10**(pKa_values['N-terminus']) / (10**(pKa_values['N-terminus']) + 10**pH)
        net_charge -= 10**pH / (10**(pKa_values['C-terminus']) + 10**pH)

        # Calculate the charge of each amino acid
        for aa in self.sequence:
            if aa in pKa_values:
                if aa in ['D', 'E', 'C', 'Y']:
                    net_charge -= 10**pH / (10**(pKa_values[aa]) + 10**pH)
                elif aa in ['H', 'K', 'R']:
                    net_charge += 10**(pKa_values[aa]) / (10**(pKa_values[aa]) + 10**pH)

        return net_charge

    def proportion_of_hydrophobic_aa(self) -> tuple:
        """
        Calculate the proportion of hydrophobic amino acids.
        :return: (Proportion of hydrophobic amino acids, Number of hydrophobic amino acids, Total number of amino acids)
        """
        hydrophobic_aas = {"G": 1, "A": 1, "V": 1, "L": 1, "I": 1, "M": 1, "F": 1, "W": 1, "P": 1,
                           "S": 0, "T": 0, "C": 0, "Y": 0, "N": 0, "Q": 0, "D": 0, "E": 0, "K": 0, "R": 0, "H": 0}
        hd_count = sum(hydrophobic_aas.get(aa, 0) for aa in self.sequence)
        total_count = len(self.sequence)
        return hd_count / total_count, hd_count, total_count

    def secondary_structure_array(self) -> list:
        """
        Calculate the proportion of helices, turns, and sheets.
        :return: List containing the proportions of helices, turns, and sheets
        """
        ss = []
        for aa in self.sequence:
            if aa in "VIYFWL":
                ss.append(1)
            elif aa in "NPGS":
                ss.append(2)
            elif aa in "EMAL":
                ss.append(3)
            else:
                ss.append(0)
        return ss

    def physical_chemical_properties(self):
        """
        Calculate the physical and chemical properties of the peptide.
        """
        self.molecular_weight = self.calculate_molecular_weight()
        self.net_charge = self.calculate_net_charge()
        self.hydrophobic_proportion = self.proportion_of_hydrophobic_aa()
        self.secondary_structure = self.secondary_structure_array()

def calculate_properties(input_csv_file, output_csv_file):
    """
    Calculate the physical and chemical properties of sequences from a CSV file and save the results to a new CSV file.
    :param input_csv_file: Path to the input CSV file containing sequences
    :param output_csv_file: Path to the output CSV file for saving results
    """
    df = pd.read_csv(input_csv_file)
    all_seq = df['Generated Sequence'].tolist()

    molecular_weights_list = []
    charges_list = []
    hydrophobicity_list = []

    for seq in all_seq:
        analysis = CustomPA(seq)
        analysis.physical_chemical_properties()
        molecular_weights_list.append(analysis.molecular_weight)
        charges_list.append(analysis.net_charge)
        hydrophobicity_list.append(analysis.hydrophobic_proportion[0])

    df_out = pd.DataFrame({
        'Sequence': all_seq,
        'Molecular Weights': molecular_weights_list,
        'Net Charges': charges_list,
        'Proportion of Hydrophobic Amino Acids': hydrophobicity_list
    })
    df_out.to_csv(output_csv_file, index=False)
    print(f"Data saved to {output_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate physical and chemical properties of sequences from a CSV file")
                        
    parser.add_argument('--input_csv_file', type=str, required=True, help="Path to the input CSV file containing sequences")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to the output CSV file for saving results")
    
    args = parser.parse_args()
    calculate_properties(args.input_csv_file, args.output_csv_file)                      
                        
                        
                        
                        
                        
                        