#!/bin/bash

WORKING_DIR=/mti881-projet2/ # Répertoire de travail
OUTPUT_DIR=/output/ # Répertoire de sortie
DATA_DIR=/data/ # Répertoire des données
ERROR_DIR = /error/ # Répertoire des erreurs


SBATCH --job-name=projet2-test-nico # Nom du job
SBATCH --mem=1G # Mémoire requise
SBATCH --gres=gpu:1 # Nombre de GPU requis - ne pas modifier !!! 
SBATCH --time=7:00:00 # Temps d'exécution demandé (heure, minutes) Ne pas dépasser 7h 

SBATCH --chdir $WORKING_DIR # Répertoire de travail
SBATCH --output %j/output.log # Fichier de sortie
SBATCH --error %j/error.log # Fichier d'erreur

# Prepare environment 
nvidia-smi
gpu 
export HF_HOME=$WORKING_DIR/hfcache
source $WORKING_DIR/venv/bin/activate

# Run task 
python3 train_conll2003.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir test-ner --do_train --do_eval --trust_remote_code=True --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --overwrite_output_dir 
