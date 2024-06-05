import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import logistic


def twfe_DGP1_simulation():
    """Perform the DGP1 simulation as per the given specifications.

    Returns:
        None: Prints the results of the simulation.

    """
    np.random.seed(42)  # You can use any integer value as the seed

    # Define parameters
    n = 1000  # Sample size
    Xsi_ps = 0.75  # pscore index
    _lambda = 0.5  # Proportion in each period

    # Define means and standard deviations
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

    # Loop for 1000 runs
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
        pi = logistic.cdf(Xsi_ps * (-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4))
        d = np.random.uniform(size=n) <= pi

        # Generate aux indexes for the potential outcomes
        index_lin = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)
        index_unobs_het = d * (index_lin)
        index_att = 0
        index_trend = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)

        # Generate unobserved heterogeneity
        v = np.random.normal(index_unobs_het, 1)

        # Generate outcomes at time 0 and time 1
        y00 = index_lin + v + np.random.normal(size=n)
        y10 = index_lin + v + np.random.normal(size=n)
        y01 = index_lin + v + np.random.normal(scale=1, size=n) + index_trend
        y11 = (
            index_lin + v + np.random.normal(scale=1, size=n) + index_trend + index_att
        )

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

        # Generate id
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
        dta_long["post:d"] = dta_long["post"] * dta_long["d"]

        dta_long = dta_long.sort_values(["id", "time"])

        # Perform TWFE estimation
        twfe_i = sm.OLS(
            dta_long["y"],
            sm.add_constant(dta_long[["x1", "x2", "x3", "x4", "post", "d", "post:d"]]),
        ).fit()
        twfe = twfe_i.params["post:d"]

        # Calculate asymptotic variance
        asymptotic_variance.append(twfe_i.cov_params()["post:d"])

        # Bootstrap for confidence intervals
        # (Note: This part is not included in the provided code, you'll need to implement it separately)

        # Append TWFE estimate to the list
        ATTE_estimates.append(twfe)

    # Convert lists to arrays for ease of calculation
    ATTE_estimates = np.array(ATTE_estimates)
    asymptotic_variance = np.array(asymptotic_variance)
    asymptotic_variance = asymptotic_variance.reshape(-1, 8)
    asymptotic_variance = asymptotic_variance[:, 0]

    # Calculate metrics
    avg_bias = np.mean(
        ATTE_estimates - 0,
    )  # Assuming ATTE is 0 as mentioned in the code
    med_bias = np.median(
        ATTE_estimates - 0,
    )  # Assuming ATTE is 0 as mentioned in the code
    rmse = np.sqrt(
        np.mean((ATTE_estimates - 0) ** 2),
    )  # Assuming ATTE is 0 as mentioned in the code
    variance_ATT = np.var(ATTE_estimates)
    coverage_probability = np.mean(
        ((ATTE_estimates - 1.96 * np.sqrt(variance_ATT)) <= 0)
        & ((ATTE_estimates + 1.96 * np.sqrt(variance_ATT)) >= 0),
    )
    avg_ci_length = np.mean(
        2 * 1.96 * np.sqrt(variance_ATT),
    )  # Length of 95% confidence interval

    return {
        "Average Bias": avg_bias,
        "Median Bias": med_bias,
        "RMSE": rmse,
        "Average Variance of ATT": variance_ATT,
        "Coverage": coverage_probability,
        "Confidence Interval Length": avg_ci_length,
    }
