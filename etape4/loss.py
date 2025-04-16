from torch import nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Le **kwargs permet d'accepter les arguments supplémentaires comme num_items_in_batch
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Vérification de sécurité
        if labels is not None and labels.max() >= len(self.class_weights):
            raise ValueError(f"Label {labels.max()} dépasse les poids disponibles ({len(self.class_weights)})")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss