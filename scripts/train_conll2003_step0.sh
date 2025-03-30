#!/bin/bash


#SBATCH --job-name=step-0-conll2003  # Nom du job
#SBATCH --mem=1G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=0:05:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h

#SBATCH --chdir=../job/step-0-conll2003  # Répertoire de travail

#SBATCH --output=%j/output.log  # Fichier de sortie
#SBATCH --error=%j/error.log  # Fichier d'erreur

echo "Working directory: $(pwd)"
echo "Starting at $(date)"
echo "Running on $(hostname)"
echo "Current user: $(whoami)"
echo "Directory ~ : $(ls ~)"

# Prepare environment
nvidia-smi

export HF_HOME=./.cache/

source ~/mti881-projet2/venv/bin/activate

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

#envoie de notification de job terminée sur discord
#mettre votre propre url de webhook (ici url du discord MTI881Projet2) : 
WEBHOOK_URL="https://discord.com/api/webhooks/1353477405624373289/-UN_D0e9qnOhK7lqVqEXLkQdCPTJn13bPNIrMn3kVRe5OfYapzS7kGN59Kn9y9mcjJcx"
MESSAGE="Job terminé : $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"$MESSAGE\"}" \
     $WEBHOOK_URL

deactivate 

