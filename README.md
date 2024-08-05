
### AMPGen: A de novo Generation Pipeline Leveraging Evolutionary Information for Broad-Spectrum Antimicrobial Peptide Design

## Overview

AMPGen is a pipeline for generating and evaluating novel antimicrobial peptide (AMP) sequences. Using EvoDiff, a novel diffusion framework in protein design, AMPGen generates new AMP sequences and employs machine learning models to classify and predict their antimicrobial efficacy. The pipeline has demonstrated exceptional success efficiency, with 29 out of 34 peptides (85.3%) exhibiting antimicrobial activity (MIC value less than 200 µg/ml against at least one bacterium).

## Methods

### Datasets Preparation

We compiled AMP and non-AMP datasets from various public databases for use in our classification and MIC prediction models. The AMP dataset includes sequences from APD, DADP, DBAASP, DRAMP, YADAMP, and dbAMP, resulting in a final set of 10,249 unique sequences with antibacterial targets. The non-AMP dataset, sourced from UniProt, consists of 11,989 sequences filtered to exclude those associated with specific antimicrobial keywords.

### De Novo AMP Generation

AMP sequences are generated using two pre-trained order-agnostic autoregressive diffusion models (OADM):

1. **Sequence-Based Generation**:
   - The sequence-based model, Evodiff-OA_DM_640M, is pre-trained on the Uniref50 dataset, containing 42 million protein sequences. This model unconditionally generates peptide sequences of length 15-35 aa.

2. **MSA-Based Generation**:
   - The MSA-based model, Evodiff-MSA_OA_DM_MAXSUB, is trained on the OpenFold dataset and generates sequences in two ways:
     - Unconditional generation of peptide sequences of length 15-35 aa.
     - Conditional generation using MSAs with known AMP sequences as representative sequences.

### Classification and Efficacy Prediction

1. **XGBoost-based AMP Classifier**:
   - The classifier is trained on features extracted from the AMP and non-AMP datasets. The final model is tuned using 10-fold cross-validation.

2. **LSTM Regression-based MIC Predictor**:
   - MIC values are predicted using separate LSTM models for Escherichia coli and Staphylococcus aureus. The datasets are split into training, validation, and test sets, and the models are trained using log-transformed MIC values.



## Project Structure

```
AMPGen/
├── AMP_classifier/
│   ├── tools/
│   ├── xgboost_model/
│   └── xgboost_results/
├── AMP_generation/
│   ├── results/
│   ├── calculate_properties.py
│   ├── conditional_generation_msa.py
│   ├── unconditional_generation.py
│   └── unconditional_generation_msa.py
├── MIC_predictor/
│   ├── lstm_model/
│   ├── tools/
│   └── main.py
├── data/
│   ├── esm_output/
│   ├── raw/
│   ├── test.csv
│   ├── top14Featured_all.csv
│   ├── 5_65_ecoli_mean_representations.csv
│   ├── 5_65_stpa_mean_representations.csv
│   ├── last5_65_ecoli_mean_representations.csv
│   ├── last5_65_stpa_mean_representations.csv
│   ├── seqs.fasta
│   ├── train5_65_ecoli_mean_representations.csv
│   └── train5_65_stpa_mean_representations.csv
├── .DS_Store
├── .gitignore
├── README.md
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Anaconda for managing environments
- Libraries: numpy, pandas, scikit-learn, xgboost, torch, biopython, evodiff

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EvoDiff-AMP.git
   cd EvoDiff-AMP
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n evodiff-amp python=3.8
   conda activate evodiff-amp
   ```

3. Install the package:
   ```bash
   pip install .
   ```

### Usage

1. **Generate New AMP Sequences**:

   - **Unconditional Generation**:
     ```bash
     unconditional_generation --total_sequences 100 --batch_size 10 --output_file /path/to/output.csv
     ```

   - **Unconditional Generation with MSA**:
     ```bash
     unconditional_generation_msa --total_sequences 100 --batch_size 10 --n_sequences 64 --output_csv_file /path/to/output.csv
     ```

   - **Conditional Generation with MSA**:
     ```bash
     conditional_generation_msa --directory_path /path/to/msa/files --output_csv_file /path/to/output.csv --max_retries 5
     ```

2. **Calculate Properties of Generated Sequences**:
   ```bash
   calculate_properties --input_csv_file /path/to/input.csv --output_csv_file /path/to/output.csv
   ```

3. **Classify and Predict Efficacy**:
   - **Train AMP Classifier**:
     ```bash
     train_amp_classifier --data_path path/to/classify_all_data_v1.csv --model_output_path path/to/save/xgboost_model.pkl
     ```

   - **Classify AMP**:
     ```bash
     classify_amp --train_path path/to/classify_all_data_v1.csv --pre_path path/to/new_sequences.csv --out_path path/to/save/predictions.csv
     ```

   - **Predict MIC Values**:
     ```bash
     predict_mic --from_csv_path path/to/input_file.csv --to_fasta_path path/to/output_fasta.fasta --esm_model_location esm2_t36_3B_UR50D --output_dir path/to/esm_output_dir --repr_layers 36 --scaler_data_path path/to/scaler.pkl --model_path path/to/model.pth --result_path path/to/result.csv
     ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

