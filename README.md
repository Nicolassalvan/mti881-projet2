# mti881-projet2

Entraînement d'un LLM sur SLURM. 



# Installation 

#  Modifications du code original

## Etape 1 : Entraînement sur MedMention 



- **Liste de labels** : Modification de la fonction `get_label_list` qui renvoit la liste des TUI (identifiants sémantiques) au format BIO, qui sont utilisés par MedMention. Renvoit la liste complète car cela change rarement, et la dimension n'augmente pas tant. Nous avons trouvé une liste sur UMLS que nous avons converti en CSV (dans `umls/tui_list.csv`). On a aussi rajouté la bibliothèque pandas aux requirements pour sa fonction `pandas.read_csv`. 

- **Script** : Modification du script, pour changer le dataset avec l'option `--dataset_name ibm-research/MedMentions-ZS`. On modifie aussi le nom du job. 


# Structure du code 

## Dossier d'entraînement (`~/job/..`)

## Dossier contenant les scripts et repo (`~/mti881_projet2`)

## Dossier `data`

Pour mettre les checkpoints, les données pour analyser l'entraînement. 





