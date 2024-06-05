"""Tasks running the core analyses."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from deep_learning_for_semiparametric_did_estimation.analysis import estimate_dr_ATTE
from deep_learning_for_semiparametric_did_estimation.config import BLD


def make_task_estimate_regression(dgp):
    def task_estimate_regression(
        did_table_year_output: Annotated[Path, Product] = BLD
        / "deep_learning_for_semiparametric_did_estimation"
        / "tables"
        / f"dr_classic_results_{dgp}.tex",
    ) -> None:
        model_results = estimate_dr_ATTE(
            n_obs=1000,
            n_rep=100,
            ATTE=0.0,
            dgp_type=dgp,
        )  # change n_rep to 100
        df = pd.DataFrame(list(model_results.items()), columns=["Measure", "Value"])
        with open(did_table_year_output, "w") as fh:
            fh.write(df.to_latex())

    return task_estimate_regression


dgp_type = [1, 2, 3, 4]

tasks = {}
for dgp in dgp_type:
    tasks[f"task_estimate_regression_{dgp}"] = make_task_estimate_regression(dgp)

# Call the tasks to execute them
for task_name, task in tasks.items():
    print(f"Running {task_name}...")
    task()
