#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:02:49 2024

@author: xiyanxiong
"""

import pandas as pd

# Load the datasets
reference_sequences_df = pd.read_csv('/Users/nicholexiong/Library/CloudStorage/OneDrive-XIYAN/Amp_model/filtered_generated_sequence/total_MSA_condition_16_filtered_2.csv')
ecoli_sequences_df = pd.read_csv('/Users/nicholexiong/Library/CloudStorage/OneDrive-XIYAN/Amp_model/filtered_generated_sequence/labelled_reference_sequence/sequences_for_escherichia_coli_test.csv')
saureus_sequences_df = pd.read_csv('/Users/nicholexiong/Library/CloudStorage/OneDrive-XIYAN/Amp_model/filtered_generated_sequence/labelled_reference_sequence/sequences_for_staphylococcus_aureus_test.csv')

# Column names (adjust if necessary)
reference_sequence_column = 'Reference Sequence'
ecoli_sequence_column = 'Sequence'
saureus_sequence_column = 'Sequence'

# Check if each reference sequence is active against E.coli
reference_sequences_df['Reference Sequence Against E.coli'] = reference_sequences_df[reference_sequence_column].isin(ecoli_sequences_df[ecoli_sequence_column])

# Check if each reference sequence is active against S.aureus
reference_sequences_df['Reference Sequence Against S.aureus'] = reference_sequences_df[reference_sequence_column].isin(saureus_sequences_df[saureus_sequence_column])

# Output the DataFrame with the new columns
print("Reference sequences with activity indicators:")
print(reference_sequences_df)

# Save the updated DataFrame to a new CSV file
reference_sequences_df.to_csv('updated_reference_total_MSA_condition_16_filtered_2.csv', index=False)







