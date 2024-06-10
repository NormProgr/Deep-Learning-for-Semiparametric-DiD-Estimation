"""Tasks running the core analyses."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from deep_learning_for_semiparametric_did_estimation.analysis import (
    het_ipw_dgp4,
    het_sz_dr_dgp4,
    ipw_sim_dgp1,
    ipw_sim_dgp2,
    ipw_sim_dgp3,
    ipw_sim_dgp4,
    sz_dr_dgp1,
    sz_dr_dgp2,
    sz_dr_dgp3,
    sz_dr_dgp4,
    twfe_DGP1_simulation,
    twfe_DGP2_simulation,
    twfe_DGP3_simulation,
    twfe_DGP4_simulation,
    het_twfe_DGP4_simulation,
)
from deep_learning_for_semiparametric_did_estimation.config import BLD

np.random.seed(42)
# TWFE tasks


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


# IPW tasks


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


# DR tasks


def task_sz_dr_dgp1(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "sz_dr_results"
    / "sz_dr_dgp1_results.tex",
) -> None:
    """Estimate the Double Robust Estimators with the DGP1."""
    model = sz_dr_dgp1()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_sz_dr_dgp2(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "sz_dr_results"
    / "sz_dr_dgp2_results.tex",
) -> None:
    """Estimate the Double Robust Estimators with the DGP2."""
    model = sz_dr_dgp2()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_sz_dr_dgp3(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "sz_dr_results"
    / "sz_dr_dgp3_results.tex",
) -> None:
    """Estimate the Double Robust Estimators with the DGP3."""
    model = sz_dr_dgp3()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_sz_dr_dgp4(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "sz_dr_results"
    / "sz_dr_dgp4_results.tex",
) -> None:
    """Estimate the IPW with the DGP4."""
    model = sz_dr_dgp4()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


# Heterogeneous Treatment Effects


def task_het_sz_dr_dgp4(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "het_results"
    / "het_sz_dr_dgp4_results.tex",
) -> None:
    """Estimate the DR DiD Estimator with the DGP4 and heterogenous treatment
    effects."""
    model = het_sz_dr_dgp4()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_het_ipw_dgp4(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "het_results"
    / "het_ipw_dgp4_results.tex",
) -> None:
    """Estimate the DR DiD Estimator with the DGP4 and heterogenous treatment
    effects."""
    model = het_ipw_dgp4()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())


def task_het_twfe_DGP4_simulation(
    did_table_year_output: Annotated[Path, Product] = BLD
    / "het_results"
    / "het_twfe_DGP4_results.tex",
) -> None:
    """Estimate the DR DiD Estimator with the DGP4 and heterogenous treatment
    effects."""
    model = het_twfe_DGP4_simulation()
    df = pd.DataFrame(list(model.items()), columns=["Measure", "Value"])
    with open(did_table_year_output, "w") as fh:
        fh.write(df.to_latex())
