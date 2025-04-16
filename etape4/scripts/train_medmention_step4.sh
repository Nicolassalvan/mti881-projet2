#!/bin/bash


#SBATCH --job-name=step-4-medmention  # Nom du job
#SBATCH --mem=8G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=07:00:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h
#SBATCH --chdir=../   # Répertoire de travail
#SBATCH --output=job/slurm-%j.out  # Fichier de sortie
#SBATCH --error=job/slurm-%j.err # Fichier d'erreur

echo "Working directory: $(pwd)"
echo "Starting at $(date)"
echo "Running on $(hostname)"
echo "Current user: $(whoami)"
echo "Directory ~ : $(ls ~)"

# Prepare environment
nvidia-smi

export HF_HOME=./.cache/


source ../venv/bin/activate

JSON_PATH="./dataset_concat.json"

# # Exécution du script
python3  ./train_medmention.py \
    --model_name_or_path bert-base-uncased \
    --train_file "$JSON_PATH" \
    --output_dir test-ner \
    --do_train \
    --do_eval \
    --do_predict \
    --trust_remote_code=True \
    --overwrite_output_dir=True \
    --save_total_limit=3 \
    --num_train_epochs=10 \
    --data_seed=42 \
    --seed=42 \
    --return_entity_level_metrics=True \
    --eval_strategy=epoch \
    --fp16 \
    --gradient_accumulation_steps=2 



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

# Envoi des résultats sur Discord
python3 ~/mti881-projet2/send_discord.py \
    --webhook_url $WEBHOOK_URL \
    --img_dir ~/mti881-projet2/etape4/figures/ 


deactivate 
