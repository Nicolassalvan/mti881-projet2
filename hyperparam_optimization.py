"""
import sys
import optuna
from train_medmention import main
import subprocess
import json
import os
from datetime import datetime
import logging
import shutil

# Configurer le logging
logging.basicConfig(level=logging.INFO, filename='hyperparam_optimization.log', filemode='w')
logger = logging.getLogger(__name__)

def objective(trial):
    try : 
        # on a un répertoire unique pour chaque test d'hyperparamètres
        trial_id = trial.number
        trial_dir = f"./checkpoints/trial_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(trial_dir, exist_ok=True)
        logger.info(f"Created directory for trial {trial_id}: {trial_dir}")

        #hyperparamètres
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        }

        # sauvegarde en JSON
        config_path = os.path.join(trial_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(hyperparams, f)

        # convertir les hyperparamètres en arguments pour le script train_medmention.py
        args = [
            "python3", "./train_medmention.py",
            "--model_name_or_path", "bert-base-uncased",
            "--dataset_name", "ibm-research/MedMentions-ZS",
            "--output_dir", trial_dir,  # utiliser le répertoire unique
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--trust_remote_code",
            "--overwrite_output_dir",
            "--load_best_model_at_end=False",
            "--resume_from_checkpoint=None",
            "--save_total_limit=3",
            "--data_seed=42",
            "--seed=42",
            "--return_entity_level_metrics=True",
            "--eval_strategy=epoch",
            "--fp16",
            "--gradient_accumulation_steps=2",
            "--config_file", config_path
        ]

        # Exécuter le script train_medmention.py avec les hyperparamètres
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        logger.info(f"Finished trial {trial_id} with output: {result.stdout}")

        #!!!!test
        if result.returncode != 0:
            logger.error(f"Training script failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError("Training script execution failed.")
        #!!!!


        files_to_keep = {"eval_results.json", "predict_results.json", "trainer_state.json", "train_results.json"}
        for filename in os.listdir(trial_dir):
            if filename not in files_to_keep:
                file_path = os.path.join(trial_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    logger.info(f"Deleted unnecessary file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}. Reason: {e}")

        #extraire la métrique d'intérêt (par exemple, la F1 score) des résultats
        try:
            metrics = json.loads(result.stdout)
            f1_score = metrics["eval_overall_f1"]
        except json.JSONDecodeError:
            f1_score = 0.0


    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        raise



def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # sauvegarder les résultats de l'optimisation
    with open("hyperparam_optimization_results.json", "w") as f:
        json.dump(study.best_trial.params, f)

if __name__ == "__main__":
    main()

"""
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
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8,16,32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
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