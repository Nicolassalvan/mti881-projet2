#!/bin/bash


#SBATCH --job-name=projet2-test-nico  # Nom du job
#SBATCH --mem=1G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=0:05:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h

#SBATCH --chdir=/srv/nfs/logti-hyper-c1/au78760@ens.ad.etsmtl.ca/job/projet2-test-nico # Répertoire de travail

#SBATCH --output=output/%j/output.log  # Fichier de sortie
#SBATCH --error=$ERROR_DIR/%j/error.log  # Fichier d'erreur

# Prepare environment
nvidia-smi
gpu
export HF_HOME=/srv/nfs/logti-hyper-c1/au78760@ens.ad.etsmtl.ca/cache
source .venv/bin/activate
pip install -r requirements.txt

# # Exécution du script
python3 train_conll2003.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name conll2003 \
    --output_dir output \
    --do_train \
    --do_eval \
    --trust_remote_code=True 

# python3 test_import.py
