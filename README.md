# EnKiteDDI

Enhanced KiteDDI with SemMedDB Features for Drug-Drug Interaction Prediction

This project provides 8-dim and 9-dim versions of the EnKiteDDI model, which integrates SemMedDB knowledge graph features to enhance drug-drug interaction prediction.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)

## Overview

EnKiteDDI is an enhanced version of KiteDDI that integrates SemMedDB knowledge graph features. This project provides two versions:

- **8-dim version**: includes has_relation + 6 relation types + normalized_count
- **9-dim version**: adds normalized_unique_pmids to the 8-dim version

### Key Features

- Integration of SemMedDB knowledge graph features
- Support for DB1 and DB2 datasets
- End-to-end training pipeline
- Comprehensive evaluation metrics

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch (>= 1.8.0)
- NumPy
- Pandas
- scikit-learn
- RDKit (rdkit-pypi)
- tqdm

### RDKit Installation

RDKit is recommended to be installed via conda:

```bash
conda install -c conda-forge rdkit
```

Or via pip:

```bash
pip install rdkit-pypi
```

### KiteDDI Dependency Modules

The training and evaluation scripts require the following Python modules from the **KiteDDI** project:

- `build_vocab.py` - Vocabulary building module (for `WordVocab` class)
- `utils.py` - Utility functions (for `split` function)
- `resnet18.py` - ResNet18 model implementation

**Download Steps:**

1. **Clone or Download KiteDDI Repository**
   - Visit: [KiteDDI GitHub Repository](https://github.com/azwad-tamir/KiteDDI)
   - Clone the repository or download the source code

2. **Copy Required Modules**
   - Copy `build_vocab.py`, `utils.py`, and `resnet18.py` from the KiteDDI repository
   - These files are typically located in the `models/KiteDDI/` directory

3. **Set Up Directory Structure**
   - Create a `KiteDDI` folder in the parent directory of this repository
   - Place the copied files (`build_vocab.py`, `utils.py`, `resnet18.py`) in the `KiteDDI/` folder
   - The directory structure should look like:
     ```
     parent_directory/
     ├── KDD/                    # This repository
     │   ├── train/
     │   ├── eval/
     │   └── ...
     └── KiteDDI/                # KiteDDI dependency modules
         ├── build_vocab.py
         ├── utils.py
         └── resnet18.py
     ```

**Note:** The scripts use `sys.path.append` to locate these modules. Make sure the `KiteDDI` folder is in the correct location relative to the script execution directory.

## Data Preparation

### 1. DB1 and DB2 Datasets

DB1 and DB2 datasets need to be downloaded from the **KiteDDI** project.

**Download Source:** [KiteDDI GitHub Repository](https://github.com/azwad-tamir/KiteDDI)

According to KiteDDI's instructions, you need to download the following data files:

#### Required Files for DB1:
- `DB1_data_allFolds` - Train/test data splits
- `db1_drugs.csv` - Drug information
- `db1_kges_transe_new.pkl` - TransE knowledge graph embeddings
- `vocab_all_smiles3.pkl` - SMILES vocabulary

#### Required Files for DB2:
- `tup_list_db2kges_transe.pkl.pkl` - Train/test data splits
- `db2_drugs.csv` - Drug information
- `db2_kges_transe_new_short.pkl` - TransE knowledge graph embeddings
- `db2_names_short.pkl` - DB2 drug names list
- `vocab_all_smiles1.pkl` - SMILES vocabulary

**Note:** Please refer to the KiteDDI project's README or download links to obtain these data files.

### 2. SemMedDB Data

SemMedDB is a database that requires permission to access.

#### Download Steps:

1. **Visit SemMedDB Official Website**
   - URL: https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB.html

2. **Apply for Data Access Permission**
   - Fill out the application form on the website
   - Wait for approval (usually takes a few business days)

3. **Download PREDICATION File**
   - Download `semmedVER43_2024_R_PREDICATION.csv` or similar version
   - The file is usually very large (several GB), ensure you have sufficient storage space

4. **Place the File**
   - Place the downloaded SemMedDB file in the `data/` directory
   - File name should be similar to: `semmedVER43_2024_R_PREDICATION.csv` or `semmedVER43_2024_R_PREDICATION.23327.csv`

### 3. Other Required Files

#### DRKG Knowledge Graph
- `drkg.tsv` - DRKG knowledge graph file
- Can be downloaded from [DRKG GitHub](https://github.com/gnn4dr/DRKG)

#### DrugBank Vocabulary (Optional, for enhanced drug name mapping)
- `drugbank vocabulary.csv` - DrugBank vocabulary
- Optional but recommended for better drug name matching

### 4. Data Directory Structure

After data preparation, your `data/` directory should contain:

```
data/
├── DB1_data_allFolds              # DB1 data splits
├── db1_drugs.csv                  # DB1 drug information
├── db1_kges_transe_new.pkl       # DB1 KG embeddings
├── vocab_all_smiles3.pkl          # DB1 vocabulary
│
├── tup_list_db2kges_transe.pkl.pkl  # DB2 data splits
├── db2_drugs.csv                     # DB2 drug information
├── db2_kges_transe_new_short.pkl    # DB2 KG embeddings
├── db2_names_short.pkl               # DB2 drug names
├── vocab_all_smiles1.pkl             # DB2 vocabulary
│
├── drkg.tsv                          # DRKG knowledge graph
├── drugbank vocabulary.csv          # DrugBank vocabulary (optional)
│
└── semmedVER43_2024_R_PREDICATION*.csv  # SemMedDB file
```

## Usage

### Step 1: Prepare SemMedDB Features

First, use the integrated script to generate SemMedDB features:

```bash
cd data_preparation
python prepare_semmeddb_features.py --variant both --db both
```

This will generate:
- `semmeddb_features_db1_8dim.pkl`
- `semmeddb_features_db1_9dim.pkl`
- `semmeddb_features_db2_8dim.pkl`
- `semmeddb_features_db2_9dim.pkl`

**Parameters:**
- `--variant both`: Generate both 8-dim and 9-dim versions
- `--db both`: Process both DB1 and DB2
- `--skip-steps 0 0b`: Skip these steps if mapping tables already exist

### Step 2: Train Models

#### DB1 Training

**8-dim version:**
```bash
cd train
python enkiteddi_db1_train_8dim.py
```

**9-dim version:**
```bash
cd train
python enkiteddi_db1_train_9dim.py
```

#### DB2 Training

**8-dim version:**
```bash
cd train
python enkiteddi_db2_train_8dim.py
```

**9-dim version:**
```bash
cd train
python enkiteddi_db2_train_9dim.py
```

### Step 3: Evaluate Models

#### DB1 Evaluation

**8-dim version:**
```bash
cd eval
python enkiteddi_db1_eval_8dim.py
```

**9-dim version:**
```bash
cd eval
python enkiteddi_db1_eval_9dim.py
```

#### DB2 Evaluation

**8-dim version:**
```bash
cd eval
python enkiteddi_db2_eval_8dim.py
```

**9-dim version:**
```bash
cd eval
python enkiteddi_db2_eval_9dim.py
```

## Model Architecture

EnKiteDDI is based on the KiteDDI architecture with the following main improvements:

1. **BERT-based Encoder**: Processes SMILES sequences
2. **ResNet18**: Extracts sequence features
3. **Knowledge Graph Embeddings**: Integrates DRKG knowledge graph embeddings
4. **SemMedDB Features**: New SemMedDB feature layer
   - 8-dim: has_relation + 6 relation types + normalized_count
   - 9-dim: 8-dim + normalized_unique_pmids

### SemMedDB Feature Description

- **has_relation**: Whether SemMedDB relation exists (0 or 1)
- **relation_types**: Binary flags for 6 relation types
  - INTERACTS_WITH
  - INHIBITS
  - STIMULATES
  - COEXISTS_WITH
  - PRODUCES
  - AFFECTS
- **normalized_count**: Normalized relation count
- **normalized_unique_pmids**: Normalized unique PMID count (9-dim only)

## Evaluation Metrics

Model evaluation includes the following metrics:

- Accuracy
- F1 Score (Micro, Macro, Weighted)
- Precision (Micro, Macro, Weighted)
- Recall (Micro, Macro, Weighted)
- MCC (Matthews Correlation Coefficient)
- AUC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)

Evaluation is performed on three data splits:
- **Validation**: Standard validation set
- **U2 (Unseen Drugs)**: Test set contains drugs not seen during training
- **U1 (Unseen Combinations)**: Test set contains drug combinations not seen during training

## Notes

1. **Data Paths**: All scripts use relative paths `../../data/`, ensure data files are in the correct location
2. **GPU Requirement**: Training requires CUDA-enabled GPU
3. **Memory Requirement**: SemMedDB processing requires large memory (recommended 16GB+)
4. **Pretrained Models**: Training automatically loads KiteDDI's pretrained BERT weights if available

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **KiteDDI**: Original KiteDDI implementation
- **SemMedDB**: For providing knowledge graph data
- **DRKG**: For providing drug knowledge graph

## Contact

For questions or suggestions, please open a GitHub Issue.
