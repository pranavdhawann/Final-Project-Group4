# Label Mappings
ID2LABEL = {0: "O", 1: "B-PII", 2: "I-PII"}
LABEL2ID = {"O": 0, "B-PII": 1, "I-PII": 2}

# Model Configuration
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LEN = 512       # Max tokens per window
STRIDE = 128        # Overlap between windows

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42
WEIGHT_DECAY = 0.01

# Data Processing
DOWNSAMPLE_CLEAN_RATIO = 0.3  # Keep 30% of non-PII documents