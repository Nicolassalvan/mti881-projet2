

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


"""
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import random
from collections import defaultdict

# 1. Configuration
MEDMENTIONS_PATH = "ibm-research/MedMentions-ZS"
CUSTOM_DATA_PATH = "./umls/curated_data_team3.json"
OUTPUT_PATH = "./medmentions_combined.json"

# 2. Chargeur robuste pour vos données
def load_custom_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    # Validation du format
    required_keys = {'tokens', 'ner_tags'}
    for i, ex in enumerate(data):
        if not all(k in ex for k in required_keys):
            raise ValueError(f"Exemple {i} manque des clés requises: {required_keys - ex.keys()}")
        if len(ex['tokens']) != len(ex['ner_tags']):
            raise ValueError(f"Exemple {i} a des tokens/tags de longueur différente")

    return data

# 3. Conversion vers un format compatible
def convert_tags(tags):
    converted = []
    for tag in tags:
        if tag == -100:
            converted.append(-100)  # On garde les -100
        elif isinstance(tag, int):
            converted.append(f"T{tag}")  # Format TXXX
        elif isinstance(tag, str):
            if tag.startswith(('B-', 'I-')) and len(tag) > 2:
                converted.append(tag)  # Tags valides
            else:
                converted.append(-100)  # Tags invalides -> ignorés
        else:
            converted.append(-100)  # Fallback
    return converted

# 4. Fonction principale
def main():
    print("1. Chargement de MedMentions...")
    try:
        medmentions = load_dataset(MEDMENTIONS_PATH, trust_remote_code=True)
        print(f"  - Exemples chargés: {len(medmentions['train'])}")
    except Exception as e:
        print(f"Erreur de chargement MedMentions: {e}")
        return

    print("\n2. Chargement des données custom...")
    try:
        custom_data = load_custom_data(CUSTOM_DATA_PATH)
        print(f"  - Exemples chargés: {len(custom_data)}")
        
        # Conversion
        converted_data = []
        for ex in custom_data:
            converted_data.append({
                'tokens': ex['tokens'],
                'ner_tags': convert_tags(ex['ner_tags']),
                'doc_id': ex.get('doc_id', 'custom')
            })
        
        custom_dataset = Dataset.from_list(converted_data)
    except Exception as e:
        print(f"Erreur de traitement des données custom: {e}")
        return

    # Vérification
    print("\n3. Vérification de compatibilité:")
    print("  - Format MedMentions:", medmentions['train'].features)
    print("  - Format Custom:", custom_dataset.features)
    
    # Stats tags
    mm_tags = set(t for ex in medmentions['train'] for t in ex['ner_tags'])
    custom_tags = set(t for ex in custom_dataset for t in ex['ner_tags'])
    print(f"\n  - Tags uniques MedMentions: {len(mm_tags)}")
    print(f"  - Tags uniques Custom: {len(custom_tags)}")
    print(f"  - Tags communs: {len(mm_tags & custom_tags)}")

    # 5. Concaténation
    print("\n4. Concaténation...")
    combined = concatenate_datasets([medmentions['train'], custom_dataset])
    print(f"  - Total exemples: {len(combined)}")
    print(f"  - Dont MedMentions: {len(medmentions['train'])}")
    print(f"  - Dont Custom: {len(custom_dataset)}")

    # 6. Test aléatoire
    print("\n5. Test aléatoire:")
    for _ in range(3):
        idx = random.randint(0, len(combined)-1)
        source = "MedMentions" if idx < len(medmentions['train']) else "Custom"
        print(f"\n  - Exemple {idx} ({source}):")
        print(f"    Tokens: {' '.join(combined[idx]['tokens'][:8])}...")
        print(f"    Tags: {combined[idx]['ner_tags'][:8]}...")

    # 7. Sauvegarde
    print("\n6. Sauvegarde...")
    combined.to_json(OUTPUT_PATH)
    print(f"  - Dataset sauvegardé dans {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
"""