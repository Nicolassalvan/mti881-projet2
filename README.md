# mti881-projet2

Entraînement d'un LLM sur SLURM. 



# Installation 

Veuillez créer un environnement virtuel dans ce répertoire avec la commande : `python -m venv ./venv`. Les bibliothèques nécessaires et leurs versions sont spécifiés dans le fichier `requirements.txt` et seront installés si nécessaires pendant le job. 

#  Modifications du code original

## Etape 1 : Entraînement sur MedMention 



- **Modification des labels** : Modification de la fonction `get_label_list` qui renvoit la liste des TUI (identifiants sémantiques) au format BIO, qui sont utilisés par MedMention. Renvoit la liste complète car cela change rarement, et la dimension n'augmente pas tant. Nous avons trouvé une liste sur UMLS que nous avons converti en CSV (dans `umls/tui_list.csv`). On a aussi rajouté la bibliothèque pandas aux requirements pour sa fonction `pandas.read_csv`. Finalement, cette méthode n'est pas très concluante et les résultats de l'évaluation sont biaisés. On va essayer de faire plutôt un mapping. Un mapping UMLS dans ce contexte permet de convertir dynamiquement les labels absents (TUIs non vus lors de l'entraînement) en leurs parents hiérarchiques les plus proches parmi les labels déjà appris par le modèle. 

- **Options du modèle** : Modification du script, pour changer le dataset avec l'option `--dataset_name ibm-research/MedMentions-ZS`. On modifie aussi le nom du job et d'autres paramètres (à décrire en s'aidant de l'aide) :

    - `lorem ipsum`
    - `lorem ipsum`
    - `lorem ipsum`


