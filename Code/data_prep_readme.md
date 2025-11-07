# NER Dataset Combiner & Synthetic Data Generator

This project combines multiple Named Entity Recognition (NER) datasets and generates synthetic data for PII (Personally Identifiable Information) detection tasks.

## Overview

The code performs three main functions:
1. **Combines multiple NER datasets** (Kaggle PII, CoNLL-2003, and PII Masking 200k) into a unified format
2. **Generates synthetic PII data** using OpenAI's GPT-4 for additional training examples
3. **Provides data analysis and visualization** of the combined dataset

## Features

### Dataset Integration
- **Kaggle PII Dataset**: Educational data with PII annotations
- **CoNLL-2003**: Standard NER benchmark with entity types mapped to PII categories
- **PII Masking 200k**: Large-scale PII dataset (English subset only)

### Synthetic Data Generation
- Creates realistic developer-related passages containing software secrets
- Supports multiple PII types: API keys, JWTs, AWS credentials, DB passwords, tokens
- Uses BIO labeling scheme for entity recognition
- Generates both original and masked versions of text

### Data Analysis
- Label frequency distribution charts
- Sentence length analysis
- PII entity proportion visualization
- Top PII entity type analysis

## Output Files

### Combined Dataset
- `Data/combined_ner_dataset.json` - JSON format with tokens and labels
- `Data/combined_ner_dataset.csv` - CSV format for easy inspection

### Synthetic Data (Optional)
- `Data/llm_generated_secrets_dataset.jsonl` - JSON Lines format with full metadata
- `Data/llm_generated_secrets_dataset.csv` - CSV with tokens and BIO labels

### Visualizations
- `Charts/label_distribution.png` - Entity label frequency
- `Charts/sentence_length_distribution.png` - Token count per sentence
- `Charts/pii_sentence_proportion.png` - PII vs non-PII sentences
- `Charts/top10_pii_labels.png` - Most common PII types

## Requirements

```bash
pip install -r requirement.txt
```

## Environment Setup

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### 1. Dataset Combination
Run the first section to combine all three datasets. This will:
- Load and preprocess each dataset
- Convert CoNLL-2003 labels to PII-compatible format
- Filter PII Masking 200k to English only
- Merge all datasets and save combined versions

### 2. Synthetic Data Generation (Optional)
Run the second section if you need additional training data. This will:
- Generate 1000 synthetic passages with PII using GPT-4
- Create proper token-level BIO labels
- Save results in both JSONL and CSV formats

### 3. Data Analysis
Run the third section to generate visualizations and statistics about:
- Label distribution across the combined dataset
- Sentence length characteristics
- PII entity proportions and frequencies

## Dataset Sources

1. **Kaggle PII Dataset**: From "pii-detection-removal-from-educational-data"
2. **CoNLL-2003**: Standard NER benchmark dataset
3. **PII Masking 200k**: Hugging Face dataset `ai4privacy/pii-masking-200k`

## Label Mapping

CoNLL-2003 entities are mapped as follows:
- `PER` → `NAME_PERSON`
- `ORG` → `ORGANIZATION` 
- `LOC` → `LOCATION`
- `MISC` → `MISC`

## Note

The synthetic  data generation section requires an OpenAI API key and will incur costs. Adjust `N_SAMPLES` as needed for your use case.