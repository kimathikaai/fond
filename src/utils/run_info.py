import wandb
from typing import List, Dict, Any, Tuple


def get_project_runs(
    api_conn: wandb.Api,
    project_name: str,
    sweep_ids: List[str],
    filtering_criteria: Dict[str, Any],
) -> Tuple[List[wandb.run], List[str]]:
    print(f"[info] Extracting sweep information from the following sweeps: {sweep_ids}")
    runs = []
    unique_datasets = []
    for sweep_id in sweep_ids:
        sweep = api_conn.sweep(f"{project_name}/{sweep_id}")
        for run in sweep.runs:
            invalid_run = False
            # remove runs that don't match all of the filteria criteria
            for filtering_key, filtering_value in filtering_criteria.items():
                if run.config[filtering_key] != filtering_value:
                    invalid_run = True
                    break
            if not invalid_run:
                if run.config["dataset"] not in unique_datasets:
                    unique_datasets.append(run.config["dataset"])
                runs.append(run)
    print(f"[info] Runs to process: {len(runs)}")
    return runs, unique_datasets


def find_best_steps(
    runs: List[wandb.run], metric_name: str, metric_goal: str, logging_freq: int = 20
):
    assert metric_goal in ["min", "max"]

    best_step_info = {}
    for i, run in enumerate(runs):
        if i != 0 and i % logging_freq == 0:
            print(f"Runs Processed: {i}")

        run_history = run.scan_history(keys=[metric_name, "step"])
        best_metric = float("inf") if metric_goal == "min" else float("-inf")
        best_step = None
        for row in run_history:
            step_metric = row[metric_name]
            if metric_goal == "min" and step_metric < best_metric:
                best_metric = step_metric
                best_step = row["step"]
            elif metric_goal == "max" and step_metric > best_metric:
                best_metric = step_metric
                best_step = row["step"]

        best_step_info[run] = {
            metric_name: best_metric,
            "step_number": best_step,
        }
    return best_step_info
