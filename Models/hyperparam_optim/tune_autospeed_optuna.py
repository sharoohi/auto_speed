from dataclasses import dataclass
from queue import Queue, Empty
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import tyro
import yaml
import torch
import json
import optuna
import traceback
import time
import threading
import shutil

from torch.utils.tensorboard import SummaryWriter
from Models.training.auto_speed_trainer import AutoSpeedTrainingArgs, train, val, Logger, new_unique_subdir


@dataclass
class AutoSpeedHPOArgs:
    """ HPO run parameters that can be set from the command line. """
    dataset: Path
    hpo_ranges: Path
    """Load HPO parameter ranges from a YAML file"""
    runs_dir: Path = Path("runs")
    optuna_db_location : Path = Path("/tmp/optuna")
    """During training, DB should be stored in fast storage to speed up execution and avoid locking problems. At the end, it will be copied to the runs dir"""
    default_config: Optional[Path] = None
    """Default config values to use for parameters not covered by the HPO parameter ranges config"""
    trials: int = 20
    epochs: int = 5
    fp32: bool = False
    """Enable full float32 precision (default: False)"""
    use_preprocessed: bool = True
    do_compile: bool = False
    do_prune: bool = False
    """Enable Optuna pruning with MedianPruner."""
    pruner_n_warmup_epochs: int = 5
    """Number of warmup epochs before MedianPruner starts pruning."""
    pruner_n_startup_trials: int = 5
    """Number of startup trials before MedianPruner starts pruning."""
    best_params_output: Optional[Path] = None
    """Path to output best params YAML file to. Default: best_params.yaml inside the run dir"""


class TrialParametersSampler:
    """
    Ranges of trial parameters specified in a YAML file.

    We can then sample from these ranges for a specific Optuna trial using
    `sample_trial_params()`.
    
    """
    def __init__(self, ranges: dict) -> None:
        self.ranges = ranges
    
    @staticmethod
    def load(yaml_file: Path) -> "TrialParametersSampler":
        with yaml_file.open("r") as f:
            return TrialParametersSampler(yaml.safe_load(f))

    def sample_trial_params(self, trial) -> dict:
        """
        Perform Optuna parameter sampling for one trial.

        This is equivalent to:

        ```
        config = {
            "min_lr": trial.suggest_loguniform("min_lr", 1e-6, 1e-4),
            ...
        }
        ```

        """
        config = {}
        for param, param_range in self.ranges.items():
            if isinstance(param_range, dict):
                if param_range["type"] == "categorical":
                    config[param] = trial.suggest_categorical(param, param_range["values"])
                else:
                    config[param] = \
                        getattr(trial, f"suggest_{param_range['type']}")(
                            param, param_range['min'], param_range['max']
                        )
            else:
                # Fixed value
                config[param] = param_range
        return config


def init_worker_process(gpu_queue: Queue):
    # This is the GPU num we will use for every job run in this worker,
    #  so each worker is strictly associated with a single GPU
    # We have to use a global to pass the GPU ID from the process init
    #  to the job
    global gpu_id
    gpu_id = gpu_queue.get()
    # Delay slightly, longer for higher GPU ids, so we don't try to init them all at once
    time.sleep(0.5 * gpu_id)
    torch.cuda.set_device(gpu_id)
    gpu_queue.task_done()


class OptimizationTrainer:
    """
    Wraps up things needed by training workers for passing conveniently into Optuna
    subprocesses.
    
    """
    def __init__(
            self, storage: str,
            hpo_args: AutoSpeedHPOArgs,
            completion_queue: Queue, 
            output_dir: Path
    ) -> None:
        self.storage = storage
        self.hpo_args = hpo_args
        self.completion_queue = completion_queue
        self.output_dir = output_dir
        # Load the parameter ranges ready to sample values for individual trials
        self.params_sampler = TrialParametersSampler.load(hpo_args.hpo_ranges)

    def train_autospeed(self, trial):
        """
        A single trial job to train for a particular set of hyperparameters.
        
        """
        # Prepare output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        run_dir = self.output_dir / f"trial_{trial.number:03d}"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(exist_ok=True, parents=True)
        log_writer = SummaryWriter(log_dir=run_dir)

        # This logger is just for logging the preparation steps, outside the train routine
        logger = Logger(run_dir / "train_process.log", log_to_stdout=False)

        # Get the GPU ID that was associated with this pool worker
        global gpu_id
        logger.write_log(f"Trial {trial.number}: assigned to GPU {gpu_id} (running on {torch.cuda.current_device()})")
        self.completion_queue.put(("running", trial.number))
        
        # Load default parameters to use for values not covered by the HPO ranges config
        if self.hpo_args.default_config is not None:
            with self.hpo_args.default_config.open("r", errors="ignore") as f:
                params = yaml.safe_load(f)
        else:
            params = {}
        # Sample hyperparameters
        config = self.params_sampler.sample_trial_params(trial)
        logger.write_log(f"Trial {trial.number} params:\n{json.dumps(config, indent=4)}")
        params.update(config)

        args = AutoSpeedTrainingArgs(
            self.hpo_args.dataset,
            batch_size=params["batch_size"],
            val_batch_size=params["val_batch_size"],
            world_size=1,
            local_rank=0,
            version="n",
            runs_dir=self.output_dir,
            epochs=self.hpo_args.epochs,
            use_preprocessed=self.hpo_args.use_preprocessed,            
            do_compile=self.hpo_args.do_compile,
            fp32=self.hpo_args.fp32,
            # Don't show progress in the individual jobs, as it gives messy output and slows us down
            disable_pbar=True,
        )

        def _epoch_callback(epoch, mean_ap, map50, recall, precision):
            self.completion_queue.put(("epoch", 1))
            trial.report(-mean_ap, epoch)
            if trial.should_prune():
                logger.write_log(
                    f"Trial {trial.number}: pruned at epoch {epoch} with mAP={mean_ap:.4g}",
                    stdout=False,
                )
                raise optuna.exceptions.TrialPruned()

        logger.write_log(f"Trial {trial.number}: calling train() for {args.epochs} epochs...")
        try:
            train(
                args, params, run_dir, log_writer, 
                log_to_stdout=False, 
                epoch_callback=_epoch_callback, 
                train_folder="train_hpo", 
                val_folder="val_hpo", 
            )
            logger.write_log(f"Trial {trial.number}: train() completed", stdout=False)
        except optuna.exceptions.TrialPruned:
            logger.write_log(f"Trial {trial.number}: pruned during training", stdout=False)
            self.completion_queue.put(("pruned", trial.number))
            raise
        except Exception as e:
            logger.write_log(f"Trial {trial.number} FAILED in train(): {e}")
            logger.write_log(traceback.format_exc())
            self.completion_queue.put(("failed", trial.number))
            raise e

        logger.write_log(f"Trial {trial.number}: calling val()...", stdout=False)
        mean_ap, map50, m_rec, m_pre = val(args, params, run_dir, log_to_stdout=False, split="val_hpo")
        logger.write_log(f"Trial {trial.number} COMPLETE. mAP={mean_ap}")
        self.completion_queue.put(("complete", trial.number))

        trial.set_user_attr("map50", map50)
        trial.set_user_attr("recall", m_rec)
        trial.set_user_attr("precision", m_pre)
        return -mean_ap
    
    def run_optimization(self, _):
        """
        Worker job for a single trial. Connects to the Optuna experiment database that
        was already created and runs one trial.

        """
        # This will load the study we created and run a single step
        job_study = optuna.load_study(storage=self.storage, study_name="autospeed_optuna")
        # Run just one trial in this job
        job_study.optimize(self.train_autospeed, n_trials=1)


if __name__ == "__main__":
    start_time = time.time()
    hpo_args : AutoSpeedHPOArgs = tyro.cli(AutoSpeedHPOArgs)

    # Get a new output subdir, so we don't overwrite old output
    output_dir = new_unique_subdir(hpo_args.runs_dir, "study")

    logger = Logger(output_dir / "optuna_trials.log", log_to_stdout=False)
    print(f"Writing logs and output to {logger.log_file_path}")

    num_gpus = torch.cuda.device_count()
    logger.write_log(f"Number of available GPUs: {num_gpus}", stdout=True)
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected. This HPO script requires at least one GPU.")

    n_trials = hpo_args.trials
    n_epochs = hpo_args.epochs
    total_epochs = n_trials * n_epochs

    # Prepare a location for the Optuna database
    # This needs to be accessible by subprocesses, but does not need to be in storage shared across nodes,
    #  as we only run on a single node. It can therefore be in fast local storage (**not** NFS!)
    hpo_args.optuna_db_location.mkdir(exist_ok=True, parents=True)
    optuna_db_path = hpo_args.optuna_db_location / "optuna_study.db"
    storage = f"sqlite:///{optuna_db_path.absolute()}"

    # Use spawn for the Optuna trial worker pool, without changing global start method.
    start_method = None  # "spawn"
    mp_ctx = mp.get_context(start_method)

    with mp_ctx.Manager() as mp_manager:
        # Reuse a single queue for epoch ticks and trial lifecycle status events.
        epoch_completion_queue = mp_manager.Queue(total_epochs + n_trials * 3)
        # Wrap up training setup in an object that we can pass into each worker
        trainer = OptimizationTrainer(storage, hpo_args, epoch_completion_queue, output_dir)
        
        study_name = "autospeed_optuna"
        if study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(
                study_name,
                storage=storage
            )
        pruner = None
        if hpo_args.do_prune:
            pruner = optuna.pruners.MedianPruner(
                n_warmup_steps=hpo_args.pruner_n_warmup_epochs,
                n_startup_trials=hpo_args.pruner_n_startup_trials,
            )

        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=study_name,
            pruner=pruner,
            load_if_exists=False,
        )
        logger.write_log(f"Created Optuna study using sampler {study.sampler.__class__.__name__}", stdout=True)

        def update_pbar():
            # Fully queue-driven status tracking avoids Optuna DB reads in the hot loop
            time.sleep(1)
                
            # Progress bar to track the total number of *epochs* completed across trials
            # If we just track completed trials, we don't see any progress for ages
            pbar = tqdm(total=total_epochs, desc="Optuna trials epochs", position=0)
            trial_states = {
                "running": 0,
                "terminated": 0,
                "pruned": 0,
            }
            seen_running_trials = set()
            seen_terminated_trials = set()
            trial_states["pending"] = n_trials
            while True:
                # Fetch progress and lifecycle events from workers.
                while True:
                    try:
                        event, value = epoch_completion_queue.get_nowait()
                    except Empty:
                        break
                    else:
                        if event == "epoch":
                            pbar.n += value
                        elif event == "running":
                            trial_num = value
                            if trial_num not in seen_running_trials and trial_num not in seen_terminated_trials:
                                seen_running_trials.add(trial_num)
                                trial_states["running"] += 1
                                trial_states["pending"] = max(0, n_trials - trial_states["running"] - trial_states["terminated"])
                        elif event in ("complete", "failed", "pruned"):
                            trial_num = value
                            if trial_num not in seen_terminated_trials:
                                seen_terminated_trials.add(trial_num)
                                if trial_num in seen_running_trials:
                                    seen_running_trials.remove(trial_num)
                                    trial_states["running"] = max(0, trial_states["running"] - 1)
                                if event == "pruned":
                                    trial_states["pruned"] += 1
                                trial_states["terminated"] += 1
                                trial_states["pending"] = max(0, n_trials - trial_states["running"] - trial_states["terminated"])
                        epoch_completion_queue.task_done()
                pbar.set_postfix({
                    "running": trial_states["running"],
                    "pending": trial_states["pending"],
                    "terminated": trial_states["terminated"],
                    "pruned": trial_states["pruned"],
                })
                pbar.refresh()
                # Exit when all trials have reported terminal status.
                if trial_states["terminated"] >= n_trials:
                    pbar.close()
                    break
                time.sleep(1)

        progress_thread = threading.Thread(target=update_pbar, daemon=True)
        progress_thread.start()

        max_parallel_jobs = min(num_gpus, n_trials)
        # These are the GPU IDs that will be shared out among the worker processes
        available_gpu_ids = mp_manager.Queue()
        for gpu_id in range(max_parallel_jobs):
            available_gpu_ids.put(gpu_id)
        
        # Start a process pool where each process is attached to a single GPU
        # Each optimization job will be run on one of these, so use its own GPU
        logger.write_log(f"Starting Optuna optimization with {max_parallel_jobs} workers", stdout=True)
        with ProcessPoolExecutor(max_parallel_jobs, mp_context=mp_ctx, initializer=init_worker_process, initargs=(available_gpu_ids,)) as pool:
            pool.map(trainer.run_optimization, range(n_trials))

        progress_thread.join()

    end_time = time.time()
    duration = end_time - start_time
    logger.write_log(f"All trials completed in {duration:.2f} seconds", stdout=True)
    logger.write_log(f"Best config: {study.best_trial.params}", stdout=True)
    logger.write_log(f"Best metrics: {study.best_trial.user_attrs}", stdout=True)

    optuna_db_output_path = output_dir / optuna_db_path.name
    logger.write_log(f"Moving Optuna db to {optuna_db_output_path}")
    shutil.move(str(optuna_db_path.absolute()), str(optuna_db_output_path.absolute()))

    best_params_yaml_path = hpo_args.best_params_output or output_dir / "best_params.yaml"
    logger.write_log(f"Writing YAML of final best params to {best_params_yaml_path}")
    with best_params_yaml_path.open("w") as f:
        yaml.dump(dict(study.best_trial.params), f)
