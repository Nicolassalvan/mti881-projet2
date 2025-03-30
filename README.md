# mti881-projet2

Entraînement d'un LLM sur SLURM. 



# Installation 

#  Modifications du code original

## Etape 1 : Entraînement sur MedMention 



- **Liste de labels** : Modification de la fonction `get_label_list` qui renvoit la liste des TUI (identifiants sémantiques) au format BIO, qui sont utilisés par MedMention. Renvoit la liste complète car cela change rarement, et la dimension n'augmente pas tant. Nous avons trouvé une liste sur UMLS que nous avons converti en CSV (dans `umls/tui_list.csv`). On a aussi rajouté la bibliothèque pandas aux requirements pour sa fonction `pandas.read_csv`. 

- **Script** : Modification du script, pour changer le dataset avec l'option `--dataset_name ibm-research/MedMentions-ZS`. On modifie aussi le nom du job. 


# Structure du code 

## Dossier d'entraînement (`~/job/..`)

Contient l'étape (0 ou 1 pour l'instant) et pour chaque étape, il y a les sous dossiers contenant les noms de jobs. Dans ces sous-dossiers, il y a les logs. 

```{bash}
~/job/ETAPE/NUMERO_JOB/output.log
~/job/ETAPE/NUMERO_JOB/error.log
```
## Dossier de code source (`~/mti881_projet2/..`)

```{bash}
~/mti881_projet2/scripts
~/mti881_projet2/umls
~/mti881_projet2/venv
```

## Dossier contenant les scripts et repo (`~/mti881_projet2/scripts`)

## Dossier `~/mti881_projet2/data`

Pour mettre les données pour analyser l'entraînement. 

## `~/mti881_projet2/venv`

`~/mti881_projet2/umls`