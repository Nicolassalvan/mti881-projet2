# mti881-projet2

Entraînement d'un LLM sur SLURM. 



# Installation 

Veuillez nommer le répertoire racine "mti881-projet2".
Veuillez créer un environnement virtuel dans ce répertoire avec la commande : `python -m venv ./venv` à la racine du projet. Les bibliothèques nécessaires et leurs versions sont spécifiés dans le fichier `requirements.txt` et seront installés si nécessaires pendant le job. 

Si vous rencontrez un problème de retour à la Ligne "\r\n" vs "\n" au moment de lancer les fichiers sh, utilisez la commande suivante : 
```{bash}
sed -i 's/\r//g' train_model_medMention.sh
```

Pour communiquer les résultats, on utilise un webhook discord. Si vous voulez utiliser cette option, il faut décommenter l'appel au script python `python3 ~/mti881-projet2/send_discord.py --webhook_url $WEBHOOK_URL --img_dir ~/mti881-projet2/etape1/figures/ ` et remplacer la variable gloabale `WEBHOOK_URL` par votre webhook. 

# Lancement du code 

Pour l'étape 1: 

```{bash}
cd ~/mti881-projet2/
sbatch etape1/train_medmention_step1.sh
```

Pour l'étape 2: 

Les fichiers curated_data_teami.json représentent les données curées parsées sous forme de json. Obtenus avec le code parse_curated_data.py (exécuté localement).
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

- **Options du modèle** : Modification du script, pour changer le dataset avec l'option `--dataset_name ibm-research/MedMentions-ZS`. On modifie aussi le nom du job.



## Etape 2 : Ajouter des données annotées du cours MTI881

Dans le code, vous pouvez observer précisément les parties qui ont été modifiées, elles sont délimitées par "#!!!".

- **Parsing des données curées avec INCEpTION** (parse_curated_data.py) : Utilisation spaCy pour tokeniser les données curées (fichiers XMI), normalisation et association des identifiants médicaux (CUIs et TUIs) via l'API UMLS, puis convertion des données en format MedMentions avec annotations BIO2, en sauvegardant le tout en JSON. Permet de fournir les fichiers curated_data_teami.json. Les entités pour lesquelles on ne trouve pas de TUI via l'API sont annotées "IGN". Utilise les fonctions de fetch_tui_from_cui.py.
  
- **Concaténation des données des équipes avec MedMention** (concat_curated_data) : Chargement et concaténation des données médicales curées avec l'ensemble de données MedMentions-ZS, en utilisant la bibliothèque datasets de Hugging Face. On lit plusieurs fichiers JSON contenant des données curées, les convertit en objets Dataset, puis les fusionne avec MedMentions. Le script vérifie la compatibilité des formats, effectue des tests aléatoires sur les données combinées et sauvegarde le résultat en JSON.

- **Modification de l'entrainement** : On annote les tokens annotés comme "IGN" en -100 pour qu'ils soient ignorés par le modèle à l'entraînement. On a aussi adapté le code à la structure de dataset_concat.json qui n'est pas délimité en train, test, eval par défaut (dans balanced_split).



## Etape 3 : Modifier l’entraînement pour utiliser un modèle génératif

Dans le code, vous pouvez observer précisément les parties qui ont été modifiées, elles sont délimitées par "#!!!".

- **Modification du modèle pré entraîné** : modification de model_name_or_path dans ModelArguments pour appeler le modèle GPT Néo 1.3B.
  
- **Adaptation du tokenizer** : GPT-Neo ne possède pas de token de padding par défaut, donc on en ajoute un pour éviter les erreurs lors du batch padding avec pad_token="<|pad|>". De plus, add_prefix_space=True permet de respecter les débuts de mots après un espace. On considérera " Hello" et "Hello" de la même façon.

- **Gestion de l'entraînement adaptée** : utilisation de accelerate permet de facilement tirer parti du multi-GPU sans avoir à gérer manuellement la distribution. Permet réduire la mémoire GPU et accélérer l'entraînement, ce qui est important pour un gros modèle comme GPT-Neo 1.3B. Configuration définie via un fichier YAML clair (accelerate_config.yaml).

## Etape 4 : Thème libre

Test de plusieurs loss pour chercher à améliorer le modèle. Utilisation d'optuna pour faire de la recherche de paramètres optimisée. 
Dans le code, vous pouvez observer précisément les parties qui ont été modifiées, elles sont délimitées par "#!!!".

-**utilisation optuna** : utiliser run_hyperparam_optimization.sh pour tester avec une loss pondérée en meme temps directement imbriquée dans le code. Le code hyperparam_optimization.py permet de paramétrer le lancement d'optuna. 

-**loss pondérée** : définie dans le fichier loss.py. Code train_medmention.py modifié pour calculer les poids par classe. 

-**focal loss** : définie dans focalLoss.py. Cette loss réduit l'influence des exemples faciles lors de l'entraînement pour mieux se concentrer sur les cas ambigus ou mal classés.

-**Adaptive Class-Balanced Focal Loss** : définie dans balancedFocalLoss.py. Cette loss combine un rééquilibrage automatique des classes rares et se focalise sur les exemples difficiles pour optimiser l'apprentissage sur des données déséquilibrées.

