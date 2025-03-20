#!/bin/bash


#SBATCH --job-name=train-0  # Nom du job
#SBATCH --mem=1G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=0:05:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h

#SBATCH --chdir=../job/train-0  # Répertoire de travail

#SBATCH --output=output/%j/output.log  # Fichier de sortie
#SBATCH --error=error/%j/error.log  # Fichier d'erreur

echo "Working directory: $(pwd)"
echo "Starting at $(date)"
echo "Running on $(hostname)"
echo "Current user: $(whoami)"
echo "Directory ~ : $(ls ~)"

# Prepare environment
nvidia-smi

export HF_HOME=./.cache/

source ../venv/bin/activate

python3 -m pip install --upgrade pip 
pip install -r ~/mti881-projet2/requirements.txt

# # Exécution du script
python3 ~/mti881-projet2/train_conll2003.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name conll2003 \
    --output_dir ../output/ \
    --do_train \
    --do_eval \
    --trust_remote_code=True 

deactivate 

