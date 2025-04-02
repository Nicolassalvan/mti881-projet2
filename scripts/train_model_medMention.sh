#!/bin/bash
#SBATCH --job-name=taskTest1
#SBATCH --output=job/slurm-%j.out
#SBATCH --error=job/slurm-%j.err
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --chdir=/srv/nfs/logti-hyper-c1/av00680@ens.ad.etsmtl.ca/projet2_phase1

nvidia-smi
export HF_HOME=/mnt/home/av00680@ens.ad.etsmtl.ca/hfcache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $TRANSFORMERS_CACHE

source /srv/nfs/logti-hyper-c1/av00680@ens.ad.etsmtl.ca/projet2_phase1/venv/bin/activate

python train_medmention.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name ibm-research/MedMentions-ZS \
  --output_dir test-ner \
  --do_train \
  --do_eval \
  --trust_remote_code=True
#mettre votre propre URL DE webhook : 
WEBHOOK_URL="https://discord.com/api/webhooks/1352871633580331078/2GwmRsUh9rtPJKc-dxBSv8ikCD-4rFmd7pck09ZTVO7P9wIi8UMx53IpXQXMMSOkkYj0"
MESSAGE="Job terminé : $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"


curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"$MESSAGE\"}" \
     $WEBHOOK_URL