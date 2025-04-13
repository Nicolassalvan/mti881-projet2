#!/bin/bash


#SBATCH --job-name=step-1-medmention  # Nom du job
#SBATCH --mem=8G  # Mémoire requise
#SBATCH --gres=gpu:1  # Nombre de GPU requis - ne pas modifier !!!
#SBATCH --time=01:00:00  # Temps d'exécution demandé (hh:mm:ss) - Ne pas dépasser 7h

#SBATCH --chdir=../job/step-1-medmention   # Répertoire de travail

#SBATCH --output=%j/output.log  # Fichier de sortie
#SBATCH --error=%j/error.log  # Fichier d'erreur

echo "Working directory: $(pwd)"
echo "Starting at $(date)"
echo "Running on $(hostname)"
echo "Current user: $(whoami)"
echo "Directory ~ : $(ls ~)"

# Prepare environment
nvidia-smi

export HF_HOME=~/mti881-projet2/.cache/

source ~/mti881-projet2/venv/bin/activate

python3 -m pip install --upgrade pip 
pip install -r ~/mti881-projet2/requirements.txt

# # Exécution du script - rajouter eventuellement max_seq_length=256 per_device_train_batch_size=8 learning_rate=1r-5 car fine-tuning 
python3  ~/mti881-projet2/etape1/train_medmention.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name ibm-research/MedMentions-ZS \
    --output_dir ~/mti881-projet2/etape1/checkpoints/$SLURM_JOB_ID/ \
    --do_train \
    --do_eval \
    --do_predict \
    --trust_remote_code=True \
    --overwrite_output_dir=True \
    --save_total_limit=3 \
    --num_train_epochs=1 \
    --data_seed=42 \
    --seed=42 \
    --return_entity_level_metrics=True \
    --eval_strategy=steps \
    --fp16 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --lr_scheduler_type=reduce_lr_on_plateau \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --max_seq_length=128 \
    



WEBHOOK_URL="https://discord.com/api/webhooks/1353477405624373289/-UN_D0e9qnOhK7lqVqEXLkQdCPTJn13bPNIrMn3kVRe5OfYapzS7kGN59Kn9y9mcjJcx"
MESSAGE="Job terminé : $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"$MESSAGE\"}" \
     $WEBHOOK_URL

# Analyse 
python3 ~/mti881-projet2/analyse_metrics.py \
    --save_dir ~/mti881-projet2/etape1/figures/ \
    --checkpoint_dir ~/mti881-projet2/etape1/checkpoints/ \

# Envoi des résultats sur Discord
python3 ~/mti881-projet2/send_discord.py \
    --webhook_url $WEBHOOK_URL \
    --img_dir ~/mti881-projet2/etape1/figures/ 

deactivate 