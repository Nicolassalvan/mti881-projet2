#!/bin/bash


#SBATCH --job-name=step-1-medmention  # Nom du job
#SBATCH --mem=8G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=01:00:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h

#SBATCH --chdir=../job/step-1-medmention   # Répertoire de travail

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
python3  ~/mti881-projet2/train_medmention.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name ibm-research/MedMentions-ZS \
    --output_dir ../output/ \
    --do_train \
    --do_eval \
    --trust_remote_code=True \
    --overwrite_output_dir=True \
    --save_total_limit=5 \
    --num_train_epochs=10 
deactivate 

