#!/bin/bash

#SBATCH --job-name=hyperparam-optuna  # Nom du job
#SBATCH --mem=8G  # MÃ©moire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=07:00:00  # Temps d'exÃ©cution demandÃ© (hh:mm:ss) - Ne pas dÃ©passer 7h
#SBATCH --chdir=../   # RÃ©pertoire de travail
#SBATCH --output=job/slurm-%j.out  # Fichier de sortie
#SBATCH --error=job/slurm-%j.err # Fichier d'erreur

echo "Working directory: $(pwd)"
echo "Starting at $(date)"
echo "Running on $(hostname)"
echo "Current user: $(whoami)"
echo "Directory ~ : $(ls ~)"

nvidia-smi

export HF_HOME=HF_HOME=./hfcache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $TRANSFORMERS_CACHE

source ../venv/bin/activate
JSON_PATH="./dataset_concat.json"

python3 ./hyperparam_optimization.py


WEBHOOK_URL="https://discord.com/api/webhooks/1353477405624373289/-UN_D0e9qnOhK7lqVqEXLkQdCPTJn13bPNIrMn3kVRe5OfYapzS7kGN59Kn9y9mcjJcx"
MESSAGE="Job terminé : $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"$MESSAGE\"}" \
     $WEBHOOK_URL

# Analyse 
echo "Analyse des résultats"

python3 ~/mti881-projet2/analyse_metrics.py \
    --save_dir ~/mti881-projet2/etape4/figures/ \
    --checkpoint_dir ~/mti881-projet2/etape4/test-ner/ \

# Envoi des résultats sur Discord
python3 ~/mti881-projet2/send_discord.py \
    --webhook_url $WEBHOOK_URL \
    --img_dir ~/mti881-projet2/etape4/figures/ 


deactivate 
