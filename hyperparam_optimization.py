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

        # hyperparamètres à optimiser
        hyperparameters = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            #"weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2), #pour l'instant test sans weight decay
        }

        # convertir les hyperparamètres en arguments pour le script train_medmention.py
        args = [
            "python3", "./train_medmention.py",
            "--model_name_or_path", "bert-base-uncased",
            "--dataset_name", "ibm-research/MedMentions-ZS",
            "--output_dir", trial_dir,  # utiliser le répertoire unique
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--trust_remote_code=True",
            "--overwrite_output_dir=True",
            "--load_best_model_at_end=False",
            "--resume_from_checkpoint=None",
            "--save_total_limit=3",
            "--data_seed=42",
            "--seed=42",
            "--return_entity_level_metrics=True",
            "--eval_strategy=epoch",
            "--fp16",
            "--gradient_accumulation_steps=2",
            "--learning_rate", str(hyperparameters["learning_rate"]),
            "--num_train_epochs", str(hyperparameters["num_train_epochs"]),
            "--per_device_train_batch_size", str(hyperparameters["batch_size"]),
        ]

        if "weight_decay" in hyperparameters:
            args.extend(["--weight_decay", str(hyperparameters["weight_decay"])])

        # Exécuter le script train_medmention.py avec les hyperparamètres
        result = subprocess.run(args, capture_output=True, text=True)
        logger.info(f"Finished trial {trial_id} with output: {result.stdout}")


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

        # extraire la métrique d'intérêt (par exemple, la F1 score) des résultats
        try:
            metrics = json.loads(result.stdout)
            f1_score = metrics["eval_overall_f1"]
        except json.JSONDecodeError:
            f1_score = 0.0

        return f1_score

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
