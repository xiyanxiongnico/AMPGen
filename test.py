#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:55:08 2024

@author: nicholexiong
"""
# tests/test_generation.py

import unittest
from src.generation.evo_diff import generate_sequences

class TestGeneration(unittest.TestCase):
    def test_generate_sequences(self):
        msa_file = 'tests/data/sample_msa.txt'
        output_dir = 'tests/output/'
        generate_sequences(msa_file, output_dir)
        # Add assertions to verify the output
        self.assertTrue(True)  # Placeholder for actual tests

if __name__ == '__main__':
    unittest.main()