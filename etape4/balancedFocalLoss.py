from torch import nn
import torch
import torch.nn.functional as F
from transformers import Trainer

class AdaptiveCBFLossTrainer(Trainer):
    def __init__(self, *args, alpha=0.75, gamma=1.0, smooth=0.05, **kwargs):
        self.class_counts = kwargs.pop('class_counts', None)
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        super().__init__(*args, **kwargs)
        
        if self.class_counts is not None:
            weights = torch.pow(1.0 / (self.class_counts + 1e-6), 0.5)
            self.class_weights = (weights / weights.sum()) * len(weights)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Reshape and filter out ignored indices (-100)
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        if active_labels.numel() == 0:  # Handle case where all labels are -100
            return (torch.tensor(0.0, device=logits.device), outputs) if return_outputs else torch.tensor(0.0, device=logits.device)

        # Label smoothing
        if self.smooth > 0:
            with torch.no_grad():
                smooth_pos = 1.0 - self.smooth
                smooth_neg = self.smooth / (model.config.num_labels - 1)
                one_hot = torch.full_like(active_logits, smooth_neg)
                one_hot.scatter_(1, active_labels.unsqueeze(1), smooth_pos)
                active_labels = one_hot

        # Compute loss
        log_pt = F.log_softmax(active_logits, dim=-1)
        
        if self.smooth > 0:
            # For label smoothing, use KL divergence
            loss = - (active_labels * log_pt).sum(dim=-1)
        else:
            # Standard cross entropy
            pt = torch.exp(log_pt)
            loss = -log_pt.gather(1, active_labels.unsqueeze(1)).squeeze(1)
            
            # Focal term
            if self.gamma > 0:
                focal_term = torch.pow(1 - pt.gather(1, active_labels.unsqueeze(1)).squeeze(1), self.gamma)
                loss = loss * focal_term

        # Class weights
        if self.class_weights is not None and self.smooth == 0:
            class_weights = self.class_weights.to(loss.device)
            loss = loss * class_weights[active_labels]

        return (loss.mean(), outputs) if return_outputs else loss.mean()