

from datasets import load_dataset, Dataset, concatenate_datasets
import json
import pandas as pd

#après avoir parsé les fichiers de données curées, on les concatene avec l'ensemble medmention


#def concat_data_with_curated():


# charger données medmentions-ZS
medmentions = load_dataset("ibm-research/MedMentions-ZS", split="train")  # ou "all", etc.

# charger les JSON donnees curees
with open("./umls/curated_data_team1.json", "r", encoding="utf-8") as f:
    data_team1 = json.load(f)

with open("./umls/curated_data_team2.json", "r", encoding="utf-8") as f:
    data_team2 = json.load(f)

with open("./umls/curated_data_team3.json", "r", encoding="utf-8") as f:
    data_team3 = json.load(f)

with open("./umls/curated_data_team4.json", "r", encoding="utf-8") as f:
    data_team4 = json.load(f)

with open("./umls/curated_data_team6.json", "r", encoding="utf-8") as f:
    data_team6 = json.load(f)

data_teams = [data_team1, data_team2, data_team3, data_team4, data_team6]

# le JSON est une liste de dicts avec "tokens", "ner_tags", et "doc_id"
# donc on enlève "doc_id"
for data in data_teams:
    for item in data:
        item.pop("doc_id", None)

# convertir en dataset huggingface
custom_dataset_team1 = Dataset.from_list(data_team1)
custom_dataset_team2 = Dataset.from_list(data_team2)
custom_dataset_team3 = Dataset.from_list(data_team3)
custom_dataset_team4 = Dataset.from_list(data_team4)
custom_dataset_team6 = Dataset.from_list(data_team6)

# concaténer
full_dataset = concatenate_datasets([medmentions, custom_dataset_team1, custom_dataset_team2, custom_dataset_team3, custom_dataset_team4, custom_dataset_team6])    

# vérification
print(full_dataset)

df = pd.DataFrame(full_dataset)

# Sauvegarder en Json
df.to_json("./umls/dataset_concat.json", orient="records", force_ascii=False, indent=2)



#verification que le fichier est bien chargé comme il faut 
data = []
with open("./umls/dataset_concat.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print("full_dataset chargé avec succès !")
print(f"Nombre d'éléments : {len(data)}")

