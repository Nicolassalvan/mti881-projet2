from torch import nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs) #initialise le trainer parent avec tous les paramètres standards (model, datasets, etc.)
        self.class_weights = class_weights  # tenseur PyTorch des poids par classe

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): #on integre la ponderation
        labels = inputs.pop("labels") #le modèle BERT attend seulement input_ids et attention_mask pour son forward pass, pas les labels
        outputs = model(**inputs) #on passe tous les éléments du dictionnaire inputs dans le modèle BERT
        logits = outputs.logits #sorties du modele avant fonction activation
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), #on transforme la forme des logits pour que chaque élément de la séquence soit comparé à une étiquette correspondante dans le calcul de la perte
                       labels.view(-1)) #on aplatie les labels de manière similaire pour qu'ils aient la même forme que les logits après redimensionnement
        
            

        return (loss, outputs) if return_outputs else loss