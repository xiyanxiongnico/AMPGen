"""
Created on Thu Jun 27 14:52:10 2024

Create a setup.py file to handle installation:
    Explanation:

	•	evodiff: This library is required for generating sequences. Ensure it is available via pip or provide installation instructions.

Additional Considerations:

	1.	Custom/Internal Libraries: If evodiff is a custom or internal library and not available via pip, you might need to provide installation instructions or include it in your repository.
	2.	Standard Libraries: Libraries like argparse, os, random, and csv are part of the Python standard library and do not need to be included in install_requires list.
    
    
@author: nicholexiong
"""

from setuptools import setup, find_packages

setup(
    name='MICPredictor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'argparse',
        'pathlib',
        'matplotlib',
        'esm',
        'scikit-learn',
        'numpy',
        'tensorflow',
        'biopython',
        'xgboost',
        'pickle5',  # pickle is included in the Python standard library, but for compatibility with different environments, you may need pickle5
        'evodiff',  # Include this if evodiff can be installed via pip
        'ifeature',  # Include this if ifeature can be installed via pip
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'predict_mic=main:main',
            'train_amp_classifier=AMP_classifier.tools.XGboost_train:main',  # Entry point for training the AMP classifier
            'classify_amp=AMP_classifier.xgboost_model.classifier:main',  # Entry point for running the AMP classifier
            'generate_unconditional_sequences=src.generation.unconditional_generation:generate_unconditional_sequences',
            'generate_unconditional_msa_sequences=src.generation.unconditional_generation_msa:generate_unconditional_msa_sequences',
            'generate_conditional_msa_sequences=src.generation.conditional_generation_msa:generate_conditional_msa_sequences',
            'calculate_properties=src.analysis.calculate_properties:calculate_properties',
        ],
    },
)