from torch import nn
import torch
import torch.nn.functional as F
from transformers import Trainer

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=0.25, gamma=2, **kwargs):
        # Extraire class_weights des kwargs si présent
        self.class_weights = kwargs.pop('class_weights', None)
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Ajout de **kwargs ici
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Vérification que class_weights est sur le bon device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        
        ce_loss = F.cross_entropy(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
            reduction='none',
            weight=self.class_weights  # Utilisation des poids si disponibles
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss