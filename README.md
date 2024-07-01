### Leveraging Generative Protein Models for the Design and Classification of Novel Antimicrobial Peptides

## Overview

This project utilizes EvoDiff, a novel diffusion framework in protein design, to generate new antimicrobial peptide (AMP) sequences. The generated sequences are then classified and evaluated for their antimicrobial efficacy using a combination of machine learning models.

## Methods

### Generation of New AMP Sequences Using EvoDiff

EvoDiff is employed to generate new AMP sequences through both unconditional and conditional generation methods:

- **Unconditional Generation**: Sequences are created from random noise by iteratively reversing the corruption process.
- **Conditional Generation**: Utilizes evolutionary information encoded in Multiple Sequence Alignments (MSAs), enabling the generation of novel sequences guided by the evolutionary context of related proteins. From each MSA, five new sequences are generated based on the reference sequence and variations guided by the alignment sequences.

### Classification and Efficacy Prediction of Generated AMP Sequences

1. **Classification**:
   - A Random Forest classifier is constructed based on feature extraction and XGBoost classification to determine whether the generated sequences are antimicrobial peptides (AMPs).

2. **Efficacy Prediction**:
   - An LSTM model utilizing language model embedding techniques is developed to evaluate the antimicrobial efficacy of the generated sequences. The performance metric used is the minimum inhibitory concentration (MIC).
   - Sequences predicted to be highly effective antimicrobial peptides are selected for experimental validation in wet lab conditions.

## Project Structure

```
AMP_Gen/
├── data/
│   ├── raw/
│   ├── processed/
├── src/
│   ├── analysis/
│   │   ├── calculate_properties.py
│   ├── classification/
│   │   ├── classifier.py
│   │   ├── features.py
│   ├── generation/
│   │   ├── unconditional_generation.py
│   │   ├── unconditional_generation_msa.py
│   │   └── conditional_generation_msa.py
│   ├── notebooks/
│   │   ├── data_analysis.ipynb
│   │   └── model_evaluation.ipynb
│   ├── results/
│   │   ├── generated_sequences/
│   │   └── model_predictions/
├── tests/
│   ├── __init__.py
│   ├── test_generation.py
│   └── # Other test scripts
├── README.md
├── requirements.txt
└── setup.py

```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Anaconda for managing environments
- Libraries: numpy, pandas, scikit-learn, xgboost, tensorflow, biopython

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

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Generate New AMP Sequences**:
   ```bash
   python src/generation/evo_diff.py
   ```

2. **Classify and Predict Efficacy**:
   ```bash
   python src/classification/random_forest_classifier.py
   python src/efficacy_prediction/lstm_model.py
   ```

3. **Analyze Results**:
   Open the Jupyter notebooks in the `notebooks/` directory to analyze data and evaluate model performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

