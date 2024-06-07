"""Tasks running the core analyses."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from deep_learning_for_semiparametric_did_estimation.analysis import (
    estimate_dr_ATTE,
    ipw_sim_dgp1,
    ipw_sim_dgp2,
    ipw_sim_dgp3,
    ipw_sim_dgp4,
    twfe_DGP1_simulation,
    twfe_DGP2_simulation,
    twfe_DGP3_simulation,
    twfe_DGP4_simulation,
)
from deep_learning_for_semiparametric_did_estimation.config import BLD

np.random.seed(42)


def task_twfe_dgp1def(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "twfe_results"
    / "twfe_dgp1_results.tex",
) -> None:
    """Estimate the Two-Way Fixed Effects model for DGP1."""
    model = twfe_DGP1_simulation()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_twfe_dgp2def(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "twfe_results"
    / "twfe_dgp2_results.tex",
) -> None:
    """Estimate the Two-Way Fixed Effects model for DGP2."""
    model = twfe_DGP2_simulation()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_twfe_dgp3def(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "twfe_results"
    / "twfe_dgp3_results.tex",
) -> None:
    """Estimate the Two-Way Fixed Effects model for DGP3."""
    model = twfe_DGP3_simulation()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_twfe_dgp4def(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "twfe_results"
    / "twfe_dgp4_results.tex",
) -> None:
    """Estimate the Two-Way Fixed Effects model for DGP4."""
    model = twfe_DGP4_simulation()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


##change here


def task_ipw_dgp1(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "ipw_results"
    / "ipw_dgp1_results.tex",
) -> None:
    """Estimate the IPW with the DGP1."""
    model = ipw_sim_dgp1()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_ipw_dgp2(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "ipw_results"
    / "ipw_dgp2_results.tex",
) -> None:
    """Estimate the IPW with the DGP2."""
    model = ipw_sim_dgp2()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_ipw_dgp3(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "ipw_results"
    / "ipw_dgp3_results.tex",
) -> None:
    """Estimate the IPW with the DGP3."""
    model = ipw_sim_dgp3()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_ipw_dgp4(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "ipw_results"
    / "ipw_dgp4_results.tex",
) -> None:
    """Estimate the IPW with the DGP4."""
    model = ipw_sim_dgp4()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def make_task_estimate_regression(dgp):
    """Estimate the Double Robust model for the given DGP type."""

    def task_estimate_regression(
        did_table_year_output: Annotated[Path, Product] = BLD
        / "dr_classic_results"
        / f"dr_classic_results_{dgp}.tex",
    ) -> None:
        # Ensure the directory exists
        did_table_year_output.parent.mkdir(parents=True, exist_ok=True)

        # Check if the output file already exists
        if not did_table_year_output.exists():
            model_results = estimate_dr_ATTE(
                n_obs=1000,
                n_rep=1000,
                ATTE=0.0,
                dgp_type=dgp,
            )
            df = pd.DataFrame(list(model_results.items()), columns=["Measure", "Value"])
            with open(did_table_year_output, "w") as fh:
                fh.write(df.to_latex())
        else:
            print(f"Output file {did_table_year_output} already exists. Skipping task.")

    return task_estimate_regression


dgp_type = [1, 2, 3, 4]

tasks = {}
for dgp in dgp_type:
    tasks[f"task_estimate_regression_{dgp}"] = make_task_estimate_regression(dgp)

# Call the tasks to execute them if necessary
for task_name, task in tasks.items():
    print(f"Running {task_name}...")
    task()
