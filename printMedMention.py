from datasets import load_dataset

# Charger MedMentions
dataset = load_dataset("ibm-research/MedMentions-ZS")

# Afficher les clés du dataset (train, validation, test)
print(dataset)

# Visualiser quelques exemples de l'ensemble d'entraînement
for example in dataset["train"].select(range(3)):  # Afficher 3 exemples
    print("\nTexte:", example["tokens"])
    print("Labels:", example["ner_tags"])
