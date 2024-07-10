import numpy as np
import pandas as pd
from scipy.special import expit as logistic_cdf
from scipy.stats import iqr, norm
from sklearn.linear_model import LogisticRegression


def std_ipw_did_rc(
    y,
    post,
    D,
    covariates=None,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=None,
    inffunc=False,
):
    # Convert inputs to numpy arrays
    D = np.asarray(D).flatten()
    n = len(D)
    y = np.asarray(y).flatten()
    post = np.asarray(post).flatten()

    # Add constant to covariate vector
    if covariates is None:
        int_cov = np.ones((n, 1))
    else:
        covariates = np.asarray(covariates)
        if np.all(covariates[:, 0] == 1):
            int_cov = covariates
        else:
            int_cov = np.hstack((np.ones((n, 1)), covariates))

    # Weights
    if i_weights is None:
        i_weights = np.ones(n)
    elif np.min(i_weights) < 0:
        msg = "i.weights must be non-negative"
        raise ValueError(msg)

    # Pscore estimation (logit) and its fitted values
    model = LogisticRegression(fit_intercept=False, solver="lbfgs")
    model.fit(int_cov, D, sample_weight=i_weights)
    ps_fit = model.predict_proba(int_cov)[:, 1]

    # Do not divide by zero
    ps_fit = np.clip(ps_fit, 1e-16, 1 - 1e-16)

    # Compute IPW estimator
    w_treat_pre = i_weights * D * (1 - post)
    w_treat_post = i_weights * D * post
    w_cont_pre = i_weights * ps_fit * (1 - D) * (1 - post) / (1 - ps_fit)
    w_cont_post = i_weights * ps_fit * (1 - D) * post / (1 - ps_fit)

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * y / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * y / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * y / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * y / np.mean(w_cont_post)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    # ATT estimator
    ipw_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre)

    # Get the influence function to compute standard error
    score_ps = i_weights.reshape(-1, 1) * (D - ps_fit).reshape(-1, 1) * int_cov
    hessian_ps = np.linalg.inv(np.dot(score_ps.T, score_ps) / n)
    asy_lin_rep_ps = np.dot(score_ps, hessian_ps)

    # Influence function of the "treat" component
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(
        w_treat_post,
    )
    inf_treat = inf_treat_post - inf_treat_pre

    # Influence function of the control component
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    # Estimation effect from gamma hat (pscore)
    M2_pre = np.mean(
        w_cont_pre.reshape(-1, 1)
        * (y - att_cont_pre).reshape(-1, 1)
        * int_cov
        / np.mean(w_cont_pre),
        axis=0,
    )
    M2_post = np.mean(
        w_cont_post.reshape(-1, 1)
        * (y - att_cont_post).reshape(-1, 1)
        * int_cov
        / np.mean(w_cont_post),
        axis=0,
    )

    inf_cont_ps = np.dot(asy_lin_rep_ps, (M2_post - M2_pre))
    inf_cont += inf_cont_ps

    # Influence function of the DR estimator
    att_inf_func = inf_treat - inf_cont

    if not boot:
        # Estimate standard error
        se_att = np.std(att_inf_func) / np.sqrt(n)
        uci = ipw_att + 1.96 * se_att
        lci = ipw_att - 1.96 * se_att
        ipw_boot = None
    else:
        if nboot is None:
            nboot = 999
        if boot_type == "multiplier":
            # Multiplier bootstrap
            multipliers = np.random.normal(size=(nboot, n))
            ipw_boot = [np.mean(m * att_inf_func) for m in multipliers]
            se_att = iqr(ipw_boot) / (norm.ppf(0.75) - norm.ppf(0.25))
            cv = np.percentile(np.abs(ipw_boot / se_att), 95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att
        else:
            # Weighted bootstrap
            ipw_boot = [
                wboot_std_ipw_rc(n, y, post, D, int_cov, i_weights)
                for _ in range(nboot)
            ]
            se_att = iqr(ipw_boot - ipw_att) / (norm.ppf(0.75) - norm.ppf(0.25))
            cv = np.percentile(np.abs((ipw_boot - ipw_att) / se_att), 95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att

    if not inffunc:
        att_inf_func = None

    return {
        "ATT": ipw_att,
        "se": se_att,
        "uci": uci,
        "lci": lci,
        "boots": ipw_boot,
        "att_inf_func": att_inf_func,
    }


def wboot_std_ipw_rc(n, y, post, D, int_cov, i_weights):
    boot_weights = np.random.choice(np.arange(1, n + 1), size=n, replace=True)
    return std_ipw_did_rc(y, post, D, int_cov, i_weights=boot_weights)["ATT"]


# New Simulation setup
def ipw_sim_dgp3():
    # Define parameters
    np.random.seed(42)

    # Sample size
    n = 1000
    # pscore index (strength of common support)
    Xsi_ps = 0.75
    # Proportion in each period
    # Number of bootstrapped draws

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
    coverage_indicators = []
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
        pi = logistic_cdf(Xsi_ps * (-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4))
        d = (np.random.uniform(size=n) <= pi).astype(int)

        # Generate aux indexes for the potential outcomes
        index_lin = 210 + 27.4 * x1 + 13.7 * (x2 + x3 + x4)

        # Create heterogeneous effects for the ATT, which is set approximately equal to zero
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
        post = (np.random.uniform(size=n) <= ti).astype(int)

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
        dta_long["post:d"] = dta_long["post"] * dta_long["d"]
        dta_long = dta_long.sort_values(["id", "time"])

        # Run the IPW-DID estimator
        covariates = dta_long[["x1", "x2", "x3", "x4"]].values
        y = dta_long["y"].values
        post = dta_long["post"].values
        D = dta_long["d"].values

        result = std_ipw_did_rc(y, post, D, covariates)

        ATTE_estimates.append(result["ATT"])
        asymptotic_variance.append(result["se"] ** 2)
        coverage_indicator = int(result["lci"] <= 0 <= result["uci"])
        coverage_indicators.append(coverage_indicator)

    # Calculate average bias, median bias, and RMSE
    true_ATT = 0

    # Bias calculations
    biases = np.array(ATTE_estimates) - true_ATT
    average_bias = np.mean(biases)
    median_bias = np.median(biases)
    average_variance = np.mean(asymptotic_variance)
    # RMSE calculation
    rmse = np.sqrt(np.mean(biases**2))
    avg_coverage_prob = np.mean(coverage_indicators)
    # Display the results
    return {
        "Average Bias": average_bias,
        "Median Bias": median_bias,
        "RMSE": rmse,
        "Average Variance of ATT": average_variance,
        "avg_coverage_prob": avg_coverage_prob,
    }
