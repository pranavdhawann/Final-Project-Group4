#%%
import json
import pandas as pd
from pathlib import Path
import os

#%%

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

# ======================
# 1️⃣ Load Kaggle PII dataset
# ======================
pii_path = Path("Data/pii-detection-removal-from-educational-data/train.json")
with open(pii_path, "r") as f:
    pii_data = json.load(f)

pii_df = pd.DataFrame([
    {"tokens": doc["tokens"], "labels": doc["labels"]}
    for doc in pii_data
])
print("✅ Loaded PII-detection dataset:", pii_df.shape)


# ======================
# 2️⃣ Load CoNLL-2003 dataset
# ======================
def load_conll(path):
    sentences, labels = [], []
    tokens, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.strip().split()
                if len(parts) == 4:
                    word, _, _, tag = parts
                    tokens.append(word)
                    # Map CoNLL labels
                    tag = tag.replace("PER", "NAME_PERSON")
                    tag = tag.replace("ORG", "ORGANIZATION")
                    tag = tag.replace("LOC", "LOCATION")
                    tag = tag.replace("MISC", "MISC")
                    tags.append(tag)
    return pd.DataFrame({"tokens": sentences, "labels": labels})

conll_df = load_conll("Data/conll2003/eng.train")
print("✅ Loaded CoNLL-2003 dataset:", conll_df.shape)


# ======================
# 3️⃣ Load PII Masking 200k (English subset only)
# ======================
print("⏳ Loading ai4privacy/pii-masking-200k...")
dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")

# Filter only English examples (column is 'language', not 'lang')
dataset_en = dataset.filter(lambda x: x["language"] == "en")

# Convert to DataFrame
pii200k_df = pd.DataFrame({
    "tokens": dataset_en["mbert_text_tokens"],   # tokens column
    "labels": dataset_en["mbert_bio_labels"]    # BIO labels column
})

print("✅ Loaded PII Masking 200k English subset:", pii200k_df.shape)


# ======================
# 4️⃣ Combine All Datasets
# ======================
combined_df = pd.concat([pii_df, conll_df, pii200k_df], ignore_index=True)
print("✅ Combined dataset shape:", combined_df.shape)


# ======================
# 5️⃣ Save Outputs
# ======================
output_dir = Path("Data")
output_dir.mkdir(exist_ok=True, parents=True)

combined_df.to_json(output_dir / "combined_ner_dataset.json", orient="records", indent=2)
combined_df.to_csv(output_dir / "combined_ner_dataset.csv", index=False)

print("✅ Combined dataset saved as:")
print("   → Data/combined_ner_dataset.json")
print("   → Data/combined_ner_dataset.csv")


#%%

"""
Only run if you want to generate synthetic data.
Also work on prompt to generate high quality dataset. Keep it <10%.

"""
import openai
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm
import pandas as pd
import csv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
openai.api_key = api_key

# Output paths
OUTPUT_DIR = Path("Data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_JSONL = OUTPUT_DIR / "llm_generated_secrets_dataset.jsonl"
OUT_CSV = OUTPUT_DIR / "llm_generated_secrets_dataset.csv"

N_SAMPLES = 1000  # number of passages to generate

# Simple whitespace tokenizer
token_split_re = re.compile(r"\S+")


def get_char_spans_for_tokens(text):
    """Return list of (token, start, end)"""
    return [(m.group(0), m.start(), m.end()) for m in token_split_re.finditer(text)]


def create_bio_labels_for_spans(token_spans, secret_spans):
    """Create BIO labels aligned to token spans"""
    labels = []
    for token, tstart, tend in token_spans:
        assigned = "O"
        for s in secret_spans:
            if not (tend <= s["start"] or tstart >= s["end"]):
                assigned = f"B-{s['label']}" if tstart == s["start"] else f"I-{s['label']}"
                break
        labels.append(assigned)
    return labels


def generate_passage():
    """Generate a passage with multiple secrets using LLM"""
    prompt = (
        "Generate a realistic developer-related passage of 50-200 words containing multiple software secrets: "
        "API keys, JWTs, AWS credentials, DB passwords, or tokens. "
        "Mask each secret with a placeholder [LABEL] for demonstration, and return the original secret in brackets after the sentence "
        "like: 'The API key is [API_KEY] (actual: ABC123)'. Make it natural and include multiple types per passage."
    )
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


def parse_passage(text):
    """Parse passage into original, masked text, and secret spans"""
    masked_text = text
    secret_spans = []
    for m in re.finditer(r"\[(\w+)\]\s*\(actual:\s*([^\)]+)\)", text):
        label, value = m.group(1), m.group(2)
        # use original text positions for token alignment
        start, end = m.start(0), m.start(0) + len(value)
        secret_spans.append({"label": label, "value": value, "start": start, "end": end})
        # replace full [LABEL] (actual: SECRET) with [LABEL] in masked text
        masked_text = masked_text.replace(m.group(0), f"[{label}]", 1)
    return text, masked_text, secret_spans


# Initialize CSV with headers
with open(OUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["tokens", "labels"])
    writer.writeheader()

# Generate dataset row by row
for i in tqdm(range(N_SAMPLES), desc="Generating samples"):
    passage = generate_passage()
    original_text, masked_text, secret_spans = parse_passage(passage)
    token_spans = get_char_spans_for_tokens(original_text)
    tokens = [t for t, s, e in token_spans]
    bio_labels = create_bio_labels_for_spans(token_spans, secret_spans)

    rec = {
        "original_text": original_text,
        "masked_text": masked_text,
        "span_labels": secret_spans,
        "tokens": tokens,
        "bio_labels": bio_labels
    }

    # Save to JSONL immediately
    with open(OUT_JSONL, "a", encoding="utf-8") as f_jsonl:
        f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save to CSV immediately
    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["tokens", "labels"])
        writer.writerow({"tokens": json.dumps(tokens, ensure_ascii=False),
                         "labels": json.dumps(bio_labels, ensure_ascii=False)})

print(f"✅ All {N_SAMPLES} samples generated and saved to {OUT_JSONL} and {OUT_CSV}")


#%%
# --- Notebook inline plotting ---
# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from pathlib import Path

# --- Load the combined dataset ---
combined_df = pd.read_json("Data/combined_ner_dataset.json")
print("✅ Combined dataset loaded!")
print("Shape:", combined_df.shape)

# --- Create Charts directory ---
charts_dir = Path("Charts")
charts_dir.mkdir(exist_ok=True)

# -----------------------------------------
# 1️⃣ Basic Info
# -----------------------------------------
print("\nNumber of sentences:", len(combined_df))
all_tokens = sum(len(tokens) for tokens in combined_df["tokens"])
print("Total tokens:", all_tokens)

# -----------------------------------------
# 2️⃣ Label Frequency Distribution
# -----------------------------------------
all_labels = [label for sublist in combined_df["labels"] for label in sublist if label != 'O']
label_counts = Counter(all_labels)
label_df = pd.DataFrame(label_counts.items(), columns=["Label", "Count"]).sort_values("Count", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(y="Label", x="Count", data=label_df, palette="crest")
plt.title("Distribution of Entity Labels")
plt.xlabel("Count")
plt.ylabel("Entity Label")
plt.tight_layout()
plt.savefig(charts_dir / "label_distribution.png", dpi=300)
plt.show()  # display in notebook
plt.close()

# -----------------------------------------
# 3️⃣ Tokens per Sentence
# -----------------------------------------
combined_df["token_count"] = combined_df["tokens"].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(combined_df["token_count"], bins=40, kde=True)
plt.title("Distribution of Sentence Lengths (Token Count)")
plt.xlabel("Tokens per Sentence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(charts_dir / "sentence_length_distribution.png", dpi=300)
plt.show()
plt.close()

# -----------------------------------------
# 4️⃣ Entity Proportion (PII vs CoNLL-style)
# -----------------------------------------
pii_entities = ["EMAIL", "USERNAME", "ID_NUM", "PHONE_NUM", "URL", "ADDRESS", "NAME_STUDENT"]
combined_df["contains_pii"] = combined_df["labels"].apply(
    lambda labels: any(any(ent in l for ent in pii_entities) for l in labels)
)

plt.figure(figsize=(6, 5))
sns.countplot(x="contains_pii", data=combined_df, palette="pastel")
plt.title("Sentences Containing PII Entities")
plt.xlabel("Contains PII Entity")
plt.ylabel("Number of Sentences")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig(charts_dir / "pii_sentence_proportion.png", dpi=300)
plt.show()
plt.close()

# -----------------------------------------
# 5️⃣ Most Common PII Entity Types (Top 10)
# -----------------------------------------
pii_labels = [label for label in all_labels if any(ent in label for ent in pii_entities)]
pii_counts = Counter(pii_labels)
pii_df_plot = pd.DataFrame(pii_counts.items(), columns=["PII_Label", "Count"]).sort_values("Count", ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(y="PII_Label", x="Count", data=pii_df_plot, palette="coolwarm")
plt.title("Top 10 PII Entity Labels")
plt.xlabel("Count")
plt.ylabel("Entity Type")
plt.tight_layout()
plt.savefig(charts_dir / "top10_pii_labels.png", dpi=300)
plt.show()
plt.close()

print(f"✅ All plots saved in '{charts_dir.resolve()}' and displayed inline!")

