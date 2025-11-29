import argparse
import torch
import gc
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# Local imports
import config
from utils import load_and_process_data, compute_metrics, tokenize_and_align_labels
from weighted_trainer import WeightedTrainer


def main(data_path, output_dir):
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load and Process Data
    df = load_and_process_data(data_path)

    # 3. Split Data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=config.RANDOM_SEED)
    train_dataset = Dataset.from_pandas(train_df[['tokens', 'unified_labels']])
    val_dataset = Dataset.from_pandas(val_df[['tokens', 'unified_labels']])

    # 4. Initialize Tokenizer
    print(f"Loading tokenizer: {config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)

    # 5. Tokenization with map
    print("Tokenizing dataset...")

    # Define a partial function or lambda to pass arguments to the mapper
    def tokenize_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer, config.MAX_LEN, config.STRIDE)

    tokenized_train = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing Train"
    )
    tokenized_val = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing Validation"
    )

    # 6. Initialize Model
    print("Initializing Model...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=3,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )
    model.to(device)

    # 7. Training Arguments
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),  # Use mixed precision if on GPU
        save_total_limit=2,  # Only keep last 2 checkpoints to save space
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # 8. Initialize Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9. Train
    print("Starting Training...")
    trainer.train()

    # 10. Save Final Model
    final_save_path = os.path.join(output_dir, "final_model")
    print(f"Saving final model to {final_save_path}...")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Training Complete.")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PII Detection Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--output_dir", type=str, default="./pii_results", help="Directory to save model checkpoints")

    args = parser.parse_args()

    main(args.data_path, args.output_dir)