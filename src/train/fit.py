import collections
import copy
import json
import time

import lightning as L
import numpy as np
import torch
import wandb

from src.datasets import DATASETS
from src.networks import ALGORITHMS
from src.utils import misc
from src.utils.hparams import default_hparams, random_hparams, seed_hash


def fit(
    seed: int,
    trial_seed: int,
    hparams_seed: int,
    algorithm_name: str,
    dataset_name: str,
    data_dir: str,
    num_workers: int,
    test_envs: list,
    overlap_type: str,
    holdout_fraction: float = 0.2,
    n_steps: int = 5001,
    checkpoint_freq: int = 300,
):
    # seed everything
    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #
    # Setup hyper parameters
    #
    if hparams_seed == 0:
        hparams = default_hparams(algorithm_name, dataset_name)
    else:
        hparams = random_hparams(
            algorithm_name, dataset_name, misc.seed_hash(hparams_seed, trial_seed)
        )

    # Add hyperparameters to config
    wandb.config.update({"hparams": hparams})

    # Get dataset
    dataset = DATASETS[dataset_name](
        root=data_dir,
        test_envs=test_envs,
        hparams=hparams,
        overlap=overlap_type,
    )

    # get overlapping classes
    hparams["C_oc"] = dataset_name.overlapping_classes

    #
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    #
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset_name):
        out, in_ = misc.split_dataset(
            env, int(len(env) * holdout_fraction), misc.seed_hash(trial_seed, env_i)
        )

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    #
    # Setup data loaders
    #
    train_loaders = [
        misc.InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in test_envs
    ]

    eval_loaders = [
        misc.FastDataLoader(
            dataset=env,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for env, _ in (in_splits + out_splits)
    ]

    eval_weights = [None for _, weights in (in_splits + out_splits)]

    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    steps_per_epoch = min([len(env) / hparams["batch_size"] for env, _ in in_splits])

    #
    # Setup the algorithm
    #
    algorithm = ALGORITHMS[algorithm_name](
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
        num_domains=len(dataset) - len(test_envs),
        hparams=hparams,
    )
    algorithm.to(device)

    #
    # Training loop
    #
    print("Begining training loop")
    checkpoint_vals = collections.defaultdict(lambda: [])

    for step in range(n_steps):
        step_start_time = time.time()

        # Get batches
        minibatches_device = [
            (x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)
        ]

        # Perform an update
        step_vals = algorithm.update(minibatches_device, None)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # log
        wandb.log({"step_time": time.time() - step_start_time})

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {}

            # Calculate training value averages
            for key, val in checkpoint_vals.items():
                if np.isnan(np.mean(val)):
                    raise Exception(f"{key}: {np.mean(val)}")
                results["train/" + str(key)] = np.mean(val)
                # tb_writer.add_scalar(key, np.mean(val), step)

            #
            # Evaluation
            #
            for name, loader, weights in zip(
                eval_loader_names,
                eval_loaders,
                eval_weights,
            ):
                (
                    acc,
                    f1,
                    overlap_class_acc,
                    non_overlap_class_acc,
                    per_class_acc,
                ) = misc.accuracy(algorithm, loader, weights, device, dataset)

                domain_idx = int(
                    name[3]
                )  # env{domain_idx}_{rest_of_string} is how name is formatted

                loader_type = ""
                if domain_idx in test_envs:
                    if "in" in name:  # Note 'in' is the 80%
                        loader_type = "test"
                    else:
                        loader_type = "ignore"
                elif "out" in name:  # Note 'out' is the 20%
                    loader_type = "val"
                else:
                    loader_type = "train"

                metric_values = {
                    loader_type + "/" + name + "_acc": float(acc),
                    loader_type + "/" + name + "_f1": float(f1),
                    loader_type + "/" + name + "_nacc": float(non_overlap_class_acc),
                    loader_type + "/" + name + "_oacc": float(overlap_class_acc),
                }
                # per class accuracy
                for class_id, class_acc in per_class_acc.items():
                    metric_values[
                        loader_type + "/" + name + "_accC" + str(class_id)
                    ] = class_acc

                results.update(metric_values)

                # log metrics
                for key, val in metric_values.items():
                    wandb.log({key: val, "step": step, "epoch": step / steps_per_epoch})

            checkpoint_vals = collections.defaultdict(lambda: [])
