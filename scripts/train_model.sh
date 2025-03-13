#!/bin/bash

# Répertoires
WORKING_DIR=/mti881-projet2  # Répertoire de travail
OUTPUT_DIR=/output  # Répertoire de sortie
DATA_DIR=/data  # Répertoire des données
ERROR_DIR=/error  # Répertoire des erreurs

# Paramètres du job
#SBATCH --job-name=projet2-test-nico  # Nom du job
#SBATCH --mem=1G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=7:00:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h
#SBATCH --chdir=$WORKING_DIR  # Répertoire de travail
#SBATCH --output=$OUTPUT_DIR/%j_output.log  # Fichier de sortie
#SBATCH --error=$ERROR_DIR/%j_error.log  # Fichier d'erreur

# Préparation de l'environnement
nvidia-smi
export HF_HOME=$WORKING_DIR/hfcache
source $WORKING_DIR/venv/bin/activate

# Exécution du script
python3 train_conll2003.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name conll2003 \
    --output_dir $OUTPUT_DIR/test-ner \
    --do_train \
    --do_eval \
    --trust_remote_code=True \
    --data_dir $DATA_DIR \
    --overwrite_output_dir
