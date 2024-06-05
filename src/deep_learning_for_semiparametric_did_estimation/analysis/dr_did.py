import numpy as np
from doubleml import DoubleMLDID
from doubleml.datasets import make_did_SZ2020
from sklearn.linear_model import LinearRegression, LogisticRegression


def estimate_dr_ATTE(n_obs, n_rep, ATTE, dgp_type):
    """Estimate the Average Treatment Effect on the Treated (ATTE) using Double Machine
    Learning Difference-in-Differences (DID) from Sant'Anna and Zhao.

    Parameters:
    n_obs (int): Number of observations for each simulation (default is 1000).
    n_rep (int): Number of replications for the simulation (default is 100).
    ATTE (float): True value of the ATTE for calculating bias and coverage (default is 0.0).

    Prints:
    - Average Bias
    - Median Bias
    - Root Mean Squared Error (RMSE)
    - Average  Variance
    - Coverage Probability
    - Average Confidence Interval Length

    """
    ml_g = LinearRegression()  # as in the paper, estimators not needed
    ml_m = LogisticRegression()  # as in the paper, estimators not needed

    ATTE_estimates = np.full((n_rep), np.nan)
    coverage = np.full((n_rep), np.nan)
    ci_length = np.full((n_rep), np.nan)
    average_variance = np.full(n_rep, np.nan)

    np.random.seed(42)
    for i_rep in range(n_rep):
        if (i_rep % int(n_rep / 10)) == 0:
            print(f"Iteration: {i_rep}/{n_rep}")
        dml_data = make_did_SZ2020(
            n_obs=n_obs,
            dgp_type=dgp_type,
            cross_sectional_data=False,
        )

        dml_did = DoubleMLDID(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=5)
        dml_did.fit()

        ATTE_estimates[i_rep] = dml_did.coef.squeeze()
        confint = dml_did.confint(level=0.95)
        coverage[i_rep] = (confint["2.5 %"].iloc[0] <= ATTE) & (
            confint["97.5 %"].iloc[0] >= ATTE
        )
        ci_length[i_rep] = confint["97.5 %"].iloc[0] - confint["2.5 %"].iloc[0]

        summary_df = dml_did.summary
        std_err = summary_df.loc["d", "std err"]
        average_variance[i_rep] = std_err**2

    # Calculate metrics
    avg_bias = np.mean(ATTE_estimates - ATTE)
    med_bias = np.median(ATTE_estimates - ATTE)
    rmse = np.sqrt(np.mean((ATTE_estimates - ATTE) ** 2))
    average_variance = np.mean(average_variance)
    coverage_probability = np.mean(coverage)
    avg_ci_length = np.mean(ci_length)

    return {
        "Average Bias": avg_bias,
        "Median Bias": med_bias,
        "RMSE": rmse,
        "Average Variance of ATT": average_variance,
        "Coverage": coverage_probability,
        "Confidence Interval Length": avg_ci_length,
    }
