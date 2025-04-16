
import sys
import optuna
import subprocess
import os
from datetime import datetime
import json
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparam_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def objective(trial):
    try:
        # Hyperparamètres
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8,16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 10),
            "weight_decay": 0.0,
        }

        # hyperparams = {
        #     "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4),  # Plage réduite
        #     "per_device_train_batch_size": 16,  # Valeur fixe (au lieu de [8,16,32])
        #     "num_train_epochs": 2,  # ← 2 époques seulement
        #     "weight_decay": 0.0,
        # }

        # Dossier de sortie
        trial_dir = f"./checkpoints/trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(trial_dir, exist_ok=True)
        logger.info(f"Début du trial {trial.number} - dossier: {trial_dir}")

        # Construction de la commande
        cmd = [
            "python", "train_medmention.py",
            "--output_dir", trial_dir,
            "--learning_rate", str(hyperparams["learning_rate"]),
            "--per_device_train_batch_size", str(hyperparams["per_device_train_batch_size"]),
            "--num_train_epochs", str(hyperparams["num_train_epochs"]),
            "--weight_decay", str(hyperparams["weight_decay"]),
            "--model_name_or_path", "bert-base-uncased",
            "--dataset_name", "ibm-research/MedMentions-ZS",
            "--do_train", "--do_eval", "--do_predict",
            "--overwrite_output_dir",
        ]

        # Exécution avec capture des logs
        logger.info(f"Exécution de la commande: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=False,  # Ne pas lever d'exception immédiatement
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Gestion des erreurs
        if result.returncode != 0:
            logger.error(f"Erreur dans le trial {trial.number}")
            logger.error(f"Code de sortie: {result.returncode}")
            logger.error(f"Sortie d'erreur:\n{result.stderr}")
            raise RuntimeError("Échec de l'exécution")

        # Récupération des métriques
        eval_file = os.path.join(trial_dir, "eval_results.json")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"Fichier {eval_file} introuvable")

        with open(eval_file) as f:
            metrics = json.load(f)

        f1_score = metrics.get("eval_f1", 0.0)
        logger.info(f"Trial {trial.number} terminé - F1-score: {f1_score}")
        return f1_score

    except Exception as e:
        logger.error(f"Erreur dans le trial {trial.number}: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50, show_progress_bar=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=True)
    
    # Sauvegarde des résultats
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f)
    
    logger.info(f"Optimisation terminée. Meilleurs paramètres: {study.best_params}")