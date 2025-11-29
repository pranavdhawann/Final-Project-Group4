import pandas as pd
import numpy as np
from ast import literal_eval
import evaluate
from config import ID2LABEL, RANDOM_SEED, DOWNSAMPLE_CLEAN_RATIO

# Load metrics once
seqeval_metric = evaluate.load("seqeval")


def unify_labels(label_list):
    """
    Maps granular classes to generic PII classes.
    O -> 0
    B-XYZ -> 1 (B-PII)
    I-XYZ -> 2 (I-PII)
    """
    unified = []
    for label in label_list:
        if label == 'O':
            unified.append(0)
        elif label.startswith('B-'):
            unified.append(1)
        else:
            unified.append(2)
    return unified


def load_and_process_data(filepath):
    """
    Loads CSV, parses lists, and downsamples clean data.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Parse columns that look like lists but are strings
    print("Parsing list columns...")
    df['tokens'] = df['tokens'].apply(literal_eval)
    df['labels'] = df['labels'].apply(literal_eval)

    # Identify rows with PII
    df['has_pii'] = df['labels'].apply(lambda x: any(l != 'O' for l in x))

    print(f"Total Documents: {len(df)}")

    # Downsampling strategy
    df_pii = df[df['has_pii']].copy()
    df_clean = df[~df['has_pii']].sample(frac=DOWNSAMPLE_CLEAN_RATIO, random_state=RANDOM_SEED).copy()

    df_balanced = pd.concat([df_pii, df_clean]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Apply unified labels
    df_balanced['unified_labels'] = df_balanced['labels'].apply(unify_labels)

    print(f"Balanced Dataset Size: {len(df_balanced)}")
    return df_balanced


def compute_metrics(p):
    """
    Computes precision, recall, f1, and accuracy using seqeval.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Filter out special tokens (-100)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(examples, tokenizer, max_len, stride):
    """
    Tokenizes inputs using sliding windows and aligns labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    labels = []
    sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")

    for i, doc_index in enumerate(sample_map):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        original_labels = examples["unified_labels"][doc_index]
        chunk_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                chunk_labels.append(-100)  # Special tokens (CLS, SEP, PAD)
            else:
                chunk_labels.append(original_labels[word_idx])

        labels.append(chunk_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs