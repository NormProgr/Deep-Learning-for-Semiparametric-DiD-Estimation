# dgp panel at work
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def twfe_DGP3_simulation():
    """Perform the DGP3 simulation as per the given specifications.

    Returns:
        None: Prints the results of the simulation.

    """
    np.random.seed(42)  # You can use any integer value as the seed

    # Sample size
    n = 1000
    # pscore index (strength of common support)
    Xsi_ps = 0.75
    # Proportion in each period
    _lambda = 0.5

    # Mean and Std deviation of Z's without truncation
    mean_z1 = np.exp(0.25 / 2)
    sd_z1 = np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
    mean_z2 = 10
    sd_z2 = 0.54164
    mean_z3 = 0.21887
    sd_z3 = 0.04453
    mean_z4 = 402
    sd_z4 = 56.63891

    # Initialize empty lists to store results
    ATTE_estimates = []
    asymptotic_variance = []
    ATTE_estimates_corr = []
    asymptotic_variance_corr = []
    coverage_prob = []
    coverage_prob_corr = []
    for _i in range(1000):
        # Generate covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        x4 = np.random.normal(0, 1, n)

        z1 = np.exp(x1 / 2)
        z2 = x2 / (1 + np.exp(x1)) + 10
        z3 = (x1 * x3 / 25 + 0.6) ** 3
        z4 = (x1 + x4 + 20) ** 2

        z1 = (z1 - mean_z1) / sd_z1
        z2 = (z2 - mean_z2) / sd_z2
        z3 = (z3 - mean_z3) / sd_z3
        z4 = (z4 - mean_z4) / sd_z4

        np.column_stack((x1, x2, x3, x4))
        np.column_stack((z1, z2, z3, z4))

        # Propensity score
        pi = 1 / (1 + np.exp(Xsi_ps * (-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4)))
        d = np.random.uniform(size=n) <= pi

        # Generate aux indexes for the potential outcomes
        index_lin = 210 + 27.4 * x1 + 13.7 * (x2 + x3 + x4)

        # Create heterogenenous effects for the ATT, which is set approximately equal to zero
        index_unobs_het = d * index_lin
        index_att = 0

        # This is the key for consistency of outcome regression
        index_trend = 210 + 27.4 * x1 + 13.7 * (x2 + x3 + x4)
        # v is the unobserved heterogeneity
        v = np.random.normal(index_unobs_het, 1)

        # Gen realized outcome at time 0
        y00 = index_lin + v + np.random.normal(size=n)
        y10 = index_lin + v + np.random.normal(size=n)

        # Gen outcomes at time 1
        y01 = (
            index_lin + v + np.random.normal(size=n) + index_trend
        )  # This is the baseline
        y11 = (
            index_lin + v + np.random.normal(size=n) + index_trend + index_att
        )  # This is the baseline

        # Generate "T"
        ti_nt = 0.5
        ti_t = 0.5
        ti = d * ti_t + (1 - d) * ti_nt
        post = np.random.uniform(size=n) <= ti

        y = np.where(
            d & post,
            y11,
            np.where(~d & post, y01, np.where(~d & ~post, y00, y10)),
        )

        # Gen id
        id_ = np.repeat(np.arange(1, n + 1), 2)
        time = np.tile([0, 1], n)

        # Put in a long data frame
        dta_long = pd.DataFrame(
            {
                "id": id_,
                "time": time,
                "y": np.tile(y, 2),
                "post": np.tile(post.astype(int), 2),
                "d": np.tile(d.astype(int), 2),
                "x1": np.tile(z1, 2),
                "x2": np.tile(z2, 2),
                "x3": np.tile(z3, 2),
                "x4": np.tile(z4, 2),
            },
        )
        dta_long = dta_long.sort_values(["id", "time"])

        # Create interaction term 'post:d'
        dta_long["post:d"] = dta_long["post"] * dta_long["d"]
        # Standard TWFE

        twfe_i = sm.OLS(
            dta_long["y"],
            sm.add_constant(dta_long[["x1", "x2", "x3", "x4", "post", "d", "post:d"]]),
        ).fit()
        twfe = twfe_i.params["post:d"]
        asymptotic_variance.append(twfe_i.cov_params().loc["post:d", "post:d"])
        lower_bound = twfe - norm.ppf(0.975) * np.sqrt(
            twfe_i.cov_params().loc["post:d", "post:d"]
        )
        upper_bound = twfe + norm.ppf(0.975) * np.sqrt(
            twfe_i.cov_params().loc["post:d", "post:d"]
        )
        coverage_prob.append(1 if lower_bound <= 0 <= upper_bound else 0)

        # Convert lists to arrays for calculations
        ATTE_estimates.append(twfe)

        # correct version of twfe
        for var in ["x1", "x2", "x3", "x4"]:
            dta_long[f"{var}:d"] = dta_long[var] * dta_long["d"]
            dta_long[f"{var}:post"] = dta_long[var] * dta_long["post"]
            dta_long[f"{var}:post:d"] = dta_long[var] * dta_long["post"] * dta_long["d"]
        independent_vars = [
            "x1",
            "x2",
            "x3",
            "x4",
            "post",
            "d",
            "post:d",
            "x1:d",
            "x2:d",
            "x3:d",
            "x4:d",
            "x1:post",
            "x2:post",
            "x3:post",
            "x4:post",
            "x1:post:d",
            "x2:post:d",
            "x3:post:d",
            "x4:post:d",
        ]
        twfe_corr_i = sm.OLS(
            dta_long["y"],
            sm.add_constant(dta_long[independent_vars]),
        ).fit()
        twfe_corr = twfe_corr_i.params["post:d"]
        asymptotic_variance_corr.append(
            twfe_corr_i.cov_params().loc["post:d", "post:d"]
        )
        ATTE_estimates_corr.append(twfe_corr)

        lower_bound_corr = twfe_corr - norm.ppf(0.975) * np.sqrt(
            twfe_corr_i.cov_params().loc["post:d", "post:d"]
        )
        upper_bound_corr = twfe_corr + norm.ppf(0.975) * np.sqrt(
            twfe_corr_i.cov_params().loc["post:d", "post:d"]
        )

        coverage_prob_corr.append(1 if lower_bound_corr <= 0 <= upper_bound_corr else 0)

    # Convert lists to arrays for ease of calculation
    ATTE_estimates = np.array(ATTE_estimates)
    asymptotic_variance = np.array(asymptotic_variance)
    asymptotic_variance = asymptotic_variance.reshape(-1, 8)
    asymptotic_variance = asymptotic_variance[:, 0]

    # Calculate metrics
    avg_bias = np.mean(ATTE_estimates - 0)
    med_bias = np.median(ATTE_estimates - 0)
    rmse = np.sqrt(
        np.mean((ATTE_estimates - 0) ** 2),
    )
    variance_ATT = np.var(ATTE_estimates)

    ATTE_estimates_corr = np.array(ATTE_estimates_corr)
    asymptotic_variance_corr = np.array(asymptotic_variance_corr)
    asymptotic_variance_corr = asymptotic_variance_corr.reshape(-1, 8)
    asymptotic_variance_corr = asymptotic_variance_corr[:, 0]
    avg_bias_corr = np.mean(
        ATTE_estimates_corr - 0,
    )
    med_bias_corr = np.median(
        ATTE_estimates_corr - 0,
    )
    rmse_corr = np.sqrt(
        np.mean((ATTE_estimates_corr - 0) ** 2),
    )
    variance_ATT_corr = np.var(ATTE_estimates_corr)
    coverage_prob = np.mean(coverage_prob)
    coverage_prob_corr = np.mean(coverage_prob_corr)
    # Print results
    return {
        "Average Bias": avg_bias,
        "Median Bias": med_bias,
        "RMSE": rmse,
        "Average Variance of ATT": variance_ATT,
        "Coverage Probability": coverage_prob,
        "Average Bias_corr": avg_bias_corr,
        "Median Bias_corr": med_bias_corr,
        "RMSE_corr": rmse_corr,
        "Average Variance of ATT_corr": variance_ATT_corr,
        "Coverage Probability_corr": coverage_prob_corr,
    }
