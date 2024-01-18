import argparse

import wandb
import yaml

from src.train.fit import fit
from src.utils.hparams import random_hparams, seed_hash

if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/extract_config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    api = wandb.Api(
        overrides={
            "entity": config["wandb"]["entity"],
        }
    )

    sweep_id = config["wandb"]["sweep_id"]
    print(f"[info] Extracting sweep information for {sweep_id}")

    sweep = api.sweep(f"{config['wandb']['project']}/{sweep_id}")
    metric_name = config["metric"]["name"]
    metric_goal = config["metric"]["goal"]
    assert metric_goal in ["min", "max"]

    # Stores best metric value and step it occurs for each run in sweep
    sweep_run_metrics = {}

    print(f"[info] Runs in sweep {sweep_id}: {len([run for run in sweep.runs])}")
    for i, run in enumerate(sweep.runs):
        if i % config["script_options"]["logging_frequency"] == 0:
            print(f"Runs Processed: {i}")

        run_id = run.id

        history = run.scan_history(keys=[metric_name, "step"])

        best_step_metric = float("inf") if metric_goal == "min" else float("-inf")
        best_step_number = None
        for row in history:
            step_metric = row[metric_name]
            if metric_goal == "min" and step_metric < best_step_metric:
                best_step_metric = step_metric
                best_step_number = row["step"]
            elif metric_goal == "max" and step_metric > best_step_metric:
                best_step_metric = step_metric
                best_step_number = row["step"]

        sweep_run_metrics[run_id] = {
            metric_name: best_step_metric,
            "step_number": best_step_number,
            "test_set_id": run.config["test_set_id"],
        }

    # Optionally outputs the best metric value and step for each run in sweep
    if config["script_options"]["display_all_runs"]:
        print(f"[info] {metric_name} for each run:")
        print(sweep_run_metrics)

    # Determine best run for sweep
    best_runs_info = {}
    for key, value in sweep_run_metrics.items():
        run_test_id = value["test_set_id"]
        run_metric_value = value[metric_name]
        if (
            (run_test_id not in best_runs_info)
            or (
                metric_goal == "min"
                and run_metric_value < best_runs_info[run_test_id][metric_name]
            )
            or (
                metric_goal == "max"
                and run_metric_value > best_runs_info[run_test_id][metric_name]
            )
        ):
            best_runs_info[value["test_set_id"]] = {
                metric_name: run_metric_value,
                "best_run_step": value["step_number"],
                "best_run": key,
            }

    for key, value in best_runs_info.items():
        print(
            f"[info] The best run for test id {key} is {value['best_run']} for with a {metric_name} of {value[metric_name]} at {value['best_run_step']}"
        )
