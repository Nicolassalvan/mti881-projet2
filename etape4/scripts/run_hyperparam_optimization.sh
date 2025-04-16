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
echo "Début de l'optimisation..."
python3 ./hyperparam_optimization.py


WEBHOOK_URL="https://discord.com/api/webhooks/1352871633580331078/2GwmRsUh9rtPJKc-dxBSv8ikCD-4rFmd7pck09ZTVO7P9wIi8UMx53IpXQXMMSOkkYj0"

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



deactivate
