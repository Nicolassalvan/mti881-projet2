# mti881-projet2

Entraînement d'un LLM sur SLURM. 



# Installation 

Veuillez créer un environnement virtuel dans ce répertoire avec la commande : `python -m venv ./venv` à la racine du projet. Les bibliothèques nécessaires et leurs versions sont spécifiés dans le fichier `requirements.txt` et seront installés si nécessaires pendant le job. 

Pour communiquer les résultats, on utilise un webhook discord. Si vous voulez utiliser cette option, il faut décommenter l'appel au script python `python3 ~/mti881-projet2/send_discord.py --webhook_url $WEBHOOK_URL --img_dir ~/mti881-projet2/etape1/figures/ ` et remplacer la variable gloabale `WEBHOOK_URL` par votre webhook. 

# Lancement du code 

Pour l'étape 1: 

```{bash}
cd ~/mti881-projet2/
sbatch etape1/train_medmention_step1.sh
```

Pour l'étape 2: 

Les fichiers curated_data_team[i].json représentent les données curées parsées sous forme de json. Obtenus avec le code parse_curated_data.py (exécuté localement).
Le fichier dataset_concat.json et la concaténation de ce fichiers avec MedMention. Obtenu grâce à concat_curated_data.py (exécuté localement).
Le code python parse_curated_data.py nécessite l'installation du modèle linguistique en_core_web_sm de spacy pour être testé, car il s’appuie sur les fonctionnalités de tokenisation et d’analyse linguistique fournies par ce modèle. Il faut ajouter sur la venv l'installation suivante : 

```{bash}
python -m spacy download en_core_web_sm
```

dataset_concat.json est utilisé en entrée du script train_medmention_step2.sh avec l'argument --train_file.

```{bash}
cd ~/mti881-projet2/
sbatch etape2/scripts/train_medmention_step2.sh
```




#  Modifications du code original

## Etape 1 : Entraînement sur MedMention 



- **Modification des labels** : Modification de la fonction `get_label_list` qui renvoit la liste des TUI (identifiants sémantiques) au format BIO, qui sont utilisés par MedMention. Renvoit la liste complète car cela change rarement, et la dimension n'augmente pas tant. Nous avons trouvé une liste sur UMLS que nous avons converti en CSV (dans `umls/tui_list.csv`). On a aussi rajouté la bibliothèque pandas aux requirements pour sa fonction `pandas.read_csv`. Finalement, cette méthode n'est pas très concluante et les résultats de l'évaluation sont biaisés. On va essayer de faire plutôt un mapping. Un mapping UMLS dans ce contexte permet de convertir dynamiquement les labels absents (TUIs non vus lors de l'entraînement) en leurs parents hiérarchiques les plus proches parmi les labels déjà appris par le modèle. 

- **Options du modèle** : Modification du script, pour changer le dataset avec l'option `--dataset_name ibm-research/MedMentions-ZS`. On modifie aussi le nom du job et d'autres paramètres (à décrire en s'aidant de l'aide) :

    - `lorem ipsum`
    - `lorem ipsum`
    - `lorem ipsum`


