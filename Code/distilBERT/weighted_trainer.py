import torch
from transformers import Trainer
import torch.nn as nn


class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to the CrossEntropyLoss.
    This helps with class imbalance (Background 'O' vs PII labels).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Retrieve device from inputs to ensure weights are on the same device as the model
        device = inputs["input_ids"].device

        # Define weights: Low weight for 'O' (0), High for 'B-PII' (1) and 'I-PII' (2)
        # Weights: [O=1.0, B-PII=15.0, I-PII=15.0]
        class_weights = torch.tensor([1.0, 15.0, 15.0]).to(device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        # Reshape logits and labels for calculation
        # Logits: (batch_size * seq_len, num_labels)
        # Labels: (batch_size * seq_len)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss