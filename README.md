
### AMPGen: A de novo Generation Pipeline Leveraging Evolutionary Information for Broad-Spectrum Antimicrobial Peptide Design

## Overview

AMPGen is a pipeline for generating and evaluating novel antimicrobial peptide (AMP) sequences. Using EvoDiff, a novel diffusion framework in protein design, AMPGen generates new AMP sequences and employs machine learning models to classify and predict their antimicrobial efficacy. The pipeline has demonstrated exceptional success efficiency, with 29 out of 34 peptides (85.3%) exhibiting antimicrobial activity (MIC value less than 200 µg/ml against at least one bacterium).

## Methods

### Datasets Preparation

We compiled AMP and non-AMP datasets from various public databases for use in our classification and MIC prediction models. The AMP dataset includes sequences from APD, DADP, DBAASP, DRAMP, YADAMP, and dbAMP, resulting in a final set of 10,249 unique sequences with antibacterial targets. The non-AMP dataset, sourced from UniProt, consists of 11,989 sequences filtered to exclude those associated with specific antimicrobial keywords.

### De Novo AMP Generation

AMP sequences are generated using two pre-trained order-agnostic autoregressive diffusion models (OADM) from the [EvoDiff framework](https://github.com/microsoft/evodiff) ([paper](https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1)).

1. **Sequence-Based Generation**:
   - The sequence-based model, Evodiff-OA_DM_640M, is pre-trained on the Uniref50 dataset, containing 42 million protein sequences. This model unconditionally generates peptide sequences of length 15-35 aa.

2. **MSA-Based Generation**:
   - The MSA-based model, Evodiff-MSA_OA_DM_MAXSUB, is trained on the OpenFold dataset and generates sequences in two ways:
     - Unconditional generation of peptide sequences of length 15-35 aa.
     - Conditional generation using MSAs with known AMP sequences as representative sequences.

### Classification and Efficacy Prediction

1. **XGBoost-based AMP Classifier**:
   - **Dataset Preparation**: Sequences in the AMP dataset were filtered based on length, retaining those within the range of 5 to 65 aa, resulting in a total of 9,964 AMP-labeled peptide sequences as the positive dataset.
   - **Feature Extraction**: Features were primarily derived from the PseKRAAC encoding method and QSOrder encoding parameters, resulting in 14 categories of 1,311 features.
   - **Model Training**: The data was used to train an XGBoost model, with AMP sequences labeled as 1 and nonAMP sequences labeled as 0. Model tuning was conducted based on the F1 score and AUC index using 10-fold cross-validation (k-fold 10) to prevent overfitting.

2. **LSTM Regression-based MIC Predictor**:
   - **Dataset Preparation**: All entries in the AMP dataset with MIC values were included. The AMP sequences targeting Escherichia coli totaled 7,100, while those targeting Staphylococcus aureus totaled 6,482. Sequences with multiple MIC values targeting the same bacteria were averaged and converted to a uniform unit of μM. These values were then log-transformed (log₁₀). Additionally, 7,193 sequences from the nonAMP dataset were labeled with a logMIC value of 4.
   - **Model Training**: Separate regression training was conducted on the Escherichia coli and Staphylococcus aureus datasets using Long Short-Term Memory (LSTM) models. The datasets were split into training, validation, and test sets in the ratio of 72:18:10. Each model comprised two LSTM layers, a dropout layer with a dropout rate of 0.7, and a linear layer. The models were compiled using standard L2 loss and optimized with the Adam optimizer.

## Project Structure

```
── AMPGen
    ├── AMP_discriminator
    │   ├── Discriminator_model
    │   │   ├── iFeature
    │   │   │   ├── codes
    │   │   │   ├── data
    │   │   │   └── PseKRAAC
    │   │   ├── discriminator.py
    │   │   └── features.py
    │   └── tools
    │       ├── plt.ipynb
    │       ├── RF_train.py
    │       ├── split.py
    │       └── XGboost_train.py
    ├── AMP_generator
    │   ├── calculate_properties.py
    │   ├── conditional_generation_msa.py
    │   ├── unconditional_generation.py
    │   └── unconditional_generation_msa.py
    ├── data
    │   ├── Discriminator_training_data
    │   │   ├── classify_all_data_v1.csv
    │   │   ├── classify_amp_v1.csv
    │   │   ├── classify_nonamp_v1.csv
    │   │   └── top14Featured_all.csv
    │   ├── Scorer_training_data
    │   │   ├── regression_ecoli_all.csv
    │   │   └── regression_stpa_all.csv
    │   ├── combined_database_filtered_v2(1).xlsx
    │   └── combined_database_v2(1).xlsx
    ├── MIC_scorer
    │   ├── results
    │   │   └── updatedecolipredictions.csv
    │   ├── Scorer_model
    │   │   ├── 1stpa_best_model_checkpoint.pth
    │   │   ├── 2ecoli_best_model_checkpoint.pth
    │   │   ├── ecoliscaler.pkl
    │   │   ├── regression.py
    │   │   └── stpascaler.pkl
    │   ├── tools
    │   │   ├── embeddingload.py
    │   │   ├── extract.py
    │   │   ├── lstm_train.py
    │   │   ├── pltlstm.ipynb
    │   │   └── tofasta.py
    │   └── scorer.py
    ├── .DS_Store
    ├── .gitattributes
    ├── .gitignore
    ├── LICENSE
    ├── print_directory_tree.py
    ├── README.md
    ├── setup.py
    └── test.py
```

## Getting Started

### Installation Guide

Welcome to the AMPGen project! This guide will walk you through the steps required to install and set up the necessary environment and dependencies to run AMPGen. Before getting started, please ensure that you have Anaconda installed on your system.

### Prerequisites

To use the AMPGen system, you need Python 3.8.5 and a few essential libraries. We'll guide you through setting up a clean conda environment, installing EvoDiff, and then the necessary dependencies. Required Python libraries: `numpy`, `pandas`, `tqdm`, `scikit-learn`, `xgboost`.

### Setting Up the Environment

1. **Clone the AMPGen Repository**  
   Begin by cloning the AMPGen repository to your local machine:
   ```bash
   git clone https://github.com/xiyanxiongnico/AMPGen.git
   cd AMPGen
   ```

2. **Create a Conda Environment**  
   Next, create a new conda environment with Python 3.8.5, which is the recommended version for this project:
   ```bash
   conda create --name AMPGen python=3.8.5
   conda activate AMPGen
   ```

3. **Install EvoDiff**  
   With the new environment activated, install the EvoDiff package, which is a crucial component of AMPGen:
   ```bash
   pip install evodiff
   ```

4. **Install PyTorch and Related Packages**  
   EvoDiff requires specific versions of PyTorch and additional libraries. Install them using the following commands:
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   conda install pyg -c pyg
   conda install -c conda-forge torch-scatter
   ```

5. **Install the Other Required Dependencies for the Discriminator and MIC Scorer**  
   The AMPGen package requires several additional dependencies to run the discriminator and MIC scorer. You can install them using the following command:
   ```bash
   pip install numpy pandas tqdm scikit-learn xgboost fair-esm matplotlib torch torchvision torchaudio pickle
   ``` 

These packages include the required libraries for both the XGBoost-based discriminator (`scikit-learn`, `xgboost`, etc.) and the MIC scorer (`torch`, `ProteinBertModel`, `MSATransformer`, `matplotlib`, etc.).

---

### Usage Guide

#### 1. **Generate New AMP Sequences**

You can generate AMP sequences using the following commands:


### Unconditional Generation of AMP Sequences

You can generate antimicrobial peptide (AMP) sequences using EvoDiff's unconditional generation model by running the following command:

```bash
python unconditional_generation.py --total_sequences <total_sequences> --batch_size <batch_size> --output_file <path_to_output_file>
```

#### Arguments:
- `--total_sequences` (int, required): The total number of sequences to generate.
- `--batch_size` (int, required): The batch size for sequence generation.
- `--output_file` (str, required): The path to the output CSV file where the generated sequences will be saved.

#### Example:
```bash
python unconditional_generation.py --total_sequences 100 --batch_size 10 --output_file ./data/generated_sequences.csv
```

This command will generate 100 sequences in batches of 10 and save them to `generated_sequences.csv`.



### Unconditional Generation of AMP Sequences with MSA

You can generate antimicrobial peptide (AMP) sequences using EvoDiff's unconditional generation model with MSA by running the following command:

```bash
python unconditional_generation_msa.py --total_sequences <total_sequences> --batch_size <batch_size> --n_sequences <n_sequences> --output_csv_file <path_to_output_file>
```

#### Arguments:
- `--total_sequences` (int, required): The total number of sequences to generate.
- `--batch_size` (int, required): The batch size for sequence generation.
- `--n_sequences` (int, required): The number of sequences in MSA to subsample.
- `--output_csv_file` (str, required): The path to the output CSV file where the generated sequences will be saved.

#### Example:
```bash
python unconditional_generation_msa.py --total_sequences 100 --batch_size 10 --n_sequences 64 --output_csv_file ./data/generated_msa_sequences.csv
```

This command will generate 100 sequences in batches of 10 and save them to `generated_msa_sequences.csv`.


### Conditional Generation of AMP Sequences with MSA

You can generate antimicrobial peptide (AMP) sequences using EvoDiff's conditional generation model with MSA by running the following command:

```bash
python conditional_generation_msa.py --directory_path <path_to_msa_directory> --output_csv_file <path_to_output_file> --max_retries <max_retries>
```

#### Arguments:
- `--directory_path` (str, required): Path to the directory containing the MSA files (in `.a3m` format).
- `--output_csv_file` (str, required): The path to the output CSV file where the generated sequences will be saved.
- `--max_retries` (int, optional): Maximum number of retries for processing each file (default: 5).

#### Example:
```bash
python conditional_generation_msa.py --directory_path ./msa_files/ --output_csv_file ./data/conditional_generated_sequences.csv --max_retries 5
```

This command will process MSA files from the `msa_files` directory and generate sequences, saving the results to `conditional_generated_sequences.csv`.


#### 2. **Calculate Properties of Generated Sequences**


### Calculate Properties of Generated AMP Sequences

You can calculate the physical and chemical properties of the generated AMP sequences, including molecular weight, net charge, and hydrophobicity, by running the following command:

```bash
python calculate_properties.py --input_csv_file <path_to_input_file> --output_csv_file <path_to_output_file>
```

#### Arguments:
- `--input_csv_file` (str, required): The path to the input CSV file containing sequences.
- `--output_csv_file` (str, required): The path to the output CSV file where the calculated properties will be saved.

#### Example:
```bash
python calculate_properties.py --input_csv_file ./data/generated_sequences.csv --output_csv_file ./data/sequence_properties.csv
```

This command will calculate the properties of the sequences in `generated_sequences.csv` and save the results to `sequence_properties.csv`.


### 3. **Identify AMP Candidates**

### AMP Discriminator

To classify sequences as antimicrobial peptides (AMPs) using the AMP Discriminator, run the following command:

```bash
python discriminator.py --train_path <path_to_training_csv> --pre_path <path_to_input_csv> --out_path <path_to_output_csv>
```

#### Arguments:
- `--train_path` or `-tp` (str, required): The path to the CSV file containing the training data.
- `--pre_path` or `-pp` (str, required): The path to the input CSV file containing the sequences to classify.
- `--out_path` or `-op` (str, required): The path to the output CSV file where the classification results will be saved.

#### Example:
```bash
python discriminator.py --train_path ./data/top14Featured_all.csv --pre_path ./data/new_sequences.csv --out_path ./results/classified_sequences.csv
```

This command will classify sequences in `new_sequences.csv` using the model trained on `classify_all_data_v1.csv`, and save the results to `classified_sequences.csv`.


### 4. **Run the MIC Scorer**

This section provides a step-by-step guide to using the MIC scorer. The scorer predicts the MIC (Minimum Inhibitory Concentration) values using a pre-trained LSTM model based on protein embeddings generated by an ESM model.

#### **Step 1: Convert Sequences to FASTA Format**
Use the `to_fasta` function to convert your input CSV file (which contains sequences) into a FASTA file:
```bash
python mic_scorer.py --from_csv_path path/to/sequences.csv --to_fasta_path path/to/output/sequences.fasta
```

#### **Step 2: Generate Embeddings with ESM Model**
Once the sequences are in FASTA format, generate their embeddings using the `get_embedding` function and a pre-trained ESM model:
```bash
python mic_scorer.py --from_csv_path path/to/sequences.csv --esm_model_location esm_model_name --output_dir path/to/output/embeddings/
```
- Replace `esm_model_name` with the location of your ESM model (e.g., `esm2_t36_3B_UR50D`).
- The embeddings will be saved to the specified output directory.

#### **Step 3: Load Embeddings**
Use the `load_embeding` function to load the generated embeddings and merge them with the input sequence data:
```bash
python mic_scorer.py --from_csv_path path/to/sequences.csv --output_dir path/to/output/embeddings/
```

#### **Step 4: Predict MIC Values**
Finally, predict MIC values using a pre-trained LSTM model. The `get_predicted_mic` function handles this task:
```bash
python mic_scorer.py --from_csv_path path/to/sequences.csv --scaler_data_path path/to/scaler.pkl --model_path path/to/model.pth --result_path path/to/save/results.csv
```

#### **Full Command Example**
Run the entire MIC scoring process from sequence conversion to MIC prediction:
```bash
python mic_scorer.py --from_csv_path path/to/sequences.csv --to_fasta_path path/to/output/sequences.fasta --esm_model_location esm2_t36_3B_UR50D --output_dir path/to/output/embeddings/ --scaler_data_path path/to/scaler.pkl --model_path path/to/model.pth --result_path path/to/save/results.csv
```

This command:
1. Converts the sequences to FASTA format.
2. Generates embeddings using the specified ESM model.
3. Loads the embeddings and prepares the data.
4. Predicts MIC values using the pre-trained LSTM model.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

