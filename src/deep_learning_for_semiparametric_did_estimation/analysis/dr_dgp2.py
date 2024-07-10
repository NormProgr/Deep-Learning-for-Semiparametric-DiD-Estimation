import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit as logistic_cdf


# Define the DRDID function
def drdid_rc(
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
    # Ensure D is a vector
    D = np.asarray(D)
    # Sample size
    n = len(D)
    # Ensure y is a vector
    y = np.asarray(y)
    # Ensure post is a vector
    post = np.asarray(post)
    # Add constant to covariate vector
    int_cov = np.ones((n, 1))
    if covariates is not None:
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

    # Compute the Pscore by MLE
    pscore_tr = sm.GLM(
        D,
        int_cov,
        family=sm.families.Binomial(),
        freq_weights=i_weights,
    ).fit()
    if not pscore_tr.converged:
        warnings.warn("GLM algorithm did not converge")
    if np.any(np.isnan(pscore_tr.params)):
        msg = "Propensity score model coefficients have NA components. Multicollinearity (or lack of variation) of covariates is a likely reason."
        raise ValueError(
            msg,
        )
    ps_fit = pscore_tr.fittedvalues
    # Avoid divide by zero
    ps_fit = np.clip(ps_fit, 1e-16, 1 - 1e-16)

    # Compute the Outcome regression for the control group at the pre-treatment period, using OLS
    reg_cont_pre = sm.WLS(
        y[(D == 0) & (post == 0)],
        int_cov[(D == 0) & (post == 0)],
        weights=i_weights[(D == 0) & (post == 0)],
    ).fit()
    if np.any(np.isnan(reg_cont_pre.params)):
        msg = "Outcome regression model coefficients have NA components. Multicollinearity (or lack of variation) of covariates is a likely reason."
        raise ValueError(
            msg,
        )
    out_y_cont_pre = int_cov @ reg_cont_pre.params

    # Compute the Outcome regression for the control group at the post-treatment period, using OLS
    reg_cont_post = sm.WLS(
        y[(D == 0) & (post == 1)],
        int_cov[(D == 0) & (post == 1)],
        weights=i_weights[(D == 0) & (post == 1)],
    ).fit()
    if np.any(np.isnan(reg_cont_post.params)):
        msg = "Outcome regression model coefficients have NA components. Multicollinearity (or lack of variation) of covariates is a likely reason."
        raise ValueError(
            msg,
        )
    out_y_cont_post = int_cov @ reg_cont_post.params

    # Combine the ORs for control group
    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # Compute the Outcome regression for the treated group at the pre-treatment period, using OLS
    reg_treat_pre = sm.WLS(
        y[(D == 1) & (post == 0)],
        int_cov[(D == 1) & (post == 0)],
        weights=i_weights[(D == 1) & (post == 0)],
    ).fit()
    out_y_treat_pre = int_cov @ reg_treat_pre.params

    # Compute the Outcome regression for the treated group at the post-treatment period, using OLS
    reg_treat_post = sm.WLS(
        y[(D == 1) & (post == 1)],
        int_cov[(D == 1) & (post == 1)],
        weights=i_weights[(D == 1) & (post == 1)],
    ).fit()
    out_y_treat_post = int_cov @ reg_treat_post.params

    # Weights
    w_treat_pre = i_weights * D * (1 - post)
    w_treat_post = i_weights * D * post
    w_cont_pre = i_weights * ps_fit * (1 - D) * (1 - post) / (1 - ps_fit)
    w_cont_post = i_weights * ps_fit * (1 - D) * post / (1 - ps_fit)

    w_d = i_weights * D
    w_dt1 = i_weights * D * post
    w_dt0 = i_weights * D * (1 - post)

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * (y - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (y - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (y - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (y - out_y_cont) / np.mean(w_cont_post)

    # Extra elements for the locally efficient DRDID
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    # ATT estimator
    dr_att = (
        (att_treat_post - att_treat_pre)
        - (att_cont_post - att_cont_pre)
        + (att_d_post - att_dt1_post)
        - (att_d_pre - att_dt0_pre)
    )

    # Get the influence function to compute standard error
    # Leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(
        w_treat_post,
    )

    # Estimation effect from beta hat from post and pre-periods
    M1_post = -np.mean(
        w_treat_post[:, np.newaxis] * post[:, np.newaxis] * int_cov,
        axis=0,
    ) / np.mean(w_treat_post)
    M1_pre = -np.mean(
        w_treat_pre[:, np.newaxis] * (1 - post)[:, np.newaxis] * int_cov,
        axis=0,
    ) / np.mean(w_treat_pre)

    # Now get the influence function related to the estimation effect related to beta's
    inf_treat_or_post = np.dot(reg_cont_post.cov_params(), M1_post)
    inf_treat_or_pre = np.dot(reg_cont_pre.cov_params(), M1_pre)
    inf_treat_or = inf_treat_or_post + inf_treat_or_pre

    # Influence function for the treated component
    inf_treat = inf_treat_post - inf_treat_pre + np.sum(inf_treat_or)

    # Now, get the influence function of control component
    # Leading term of the influence function: no estimation effect from nuisance parameters
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

    # Estimation effect from gamma hat (pscore)
    M2_pre = np.mean(
        w_cont_pre[:, np.newaxis]
        * (y[:, np.newaxis] - out_y_cont[:, np.newaxis] - att_cont_pre)
        * int_cov,
        axis=0,
    ) / np.mean(w_cont_pre)
    M2_post = np.mean(
        w_cont_post[:, np.newaxis]
        * (y[:, np.newaxis] - out_y_cont[:, np.newaxis] - att_cont_post)
        * int_cov,
        axis=0,
    ) / np.mean(w_cont_post)

    # Now the influence function related to estimation effect of pscores
    inf_cont_ps = np.dot(pscore_tr.cov_params(), (M2_post - M2_pre))
    inf_cont_ps = np.sum(inf_cont_ps)

    # Estimation effect from beta hat from post and pre-periods
    M3_post = -np.mean(
        w_cont_post[:, np.newaxis] * post[:, np.newaxis] * int_cov,
        axis=0,
    ) / np.mean(w_cont_post)
    M3_pre = -np.mean(
        w_cont_pre[:, np.newaxis] * (1 - post)[:, np.newaxis] * int_cov,
        axis=0,
    ) / np.mean(w_cont_pre)

    # Now get the influence function related to the estimation effect related to beta's
    inf_cont_or_post = np.dot(reg_cont_post.cov_params(), M3_post)
    inf_cont_or_pre = np.dot(reg_cont_pre.cov_params(), M3_pre)
    inf_cont_or = inf_cont_or_post + inf_cont_or_pre
    inf_cont_or = np.sum(inf_cont_or)

    # Influence function for the control component
    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    # Get the influence function of the inefficient DR estimator (put all pieces together)
    dr_att_inf_func1 = inf_treat - inf_cont

    # Now, we only need to get the influence function of the adjustment terms
    # First, the terms as if all OR parameters were known
    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # Now the estimation effect of the OR coefficients
    mom_post = np.mean(
        (w_d / np.mean(w_d) - w_dt1 / np.mean(w_dt1))[:, np.newaxis] * int_cov,
        axis=0,
    )
    mom_pre = np.mean(
        (w_d / np.mean(w_d) - w_dt0 / np.mean(w_dt0))[:, np.newaxis] * int_cov,
        axis=0,
    )
    inf_or_post = np.dot(
        (reg_treat_post.cov_params() - reg_cont_post.cov_params()),
        mom_post,
    )
    inf_or_pre = np.dot(
        (reg_treat_pre.cov_params() - reg_cont_pre.cov_params()),
        mom_pre,
    )
    inf_or = inf_or_post - inf_or_pre
    inf_or = np.sum(inf_or)

    # Get the influence function of the locally efficient DR estimator (put all pieces together)
    dr_att_inf_func = dr_att_inf_func1 + inf_eff + inf_or

    if not boot:
        # Estimate of standard error
        se_dr_att = np.std(dr_att_inf_func) / np.sqrt(n)
        # Estimate of upper boundary of 95% CI
        uci = dr_att + 1.96 * se_dr_att
        # Estimate of lower boundary of 95% CI
        lci = dr_att - 1.96 * se_dr_att
        # Create this null vector so we can export the bootstrap draws too.
        dr_boot = None
    else:
        if nboot is None:
            nboot = 999
        if boot_type == "multiplier":
            # Do multiplier bootstrap
            dr_boot = mboot_did(dr_att_inf_func, nboot)
            # Get bootstrap std errors based on IQR
            se_dr_att = np.percentile(dr_boot, 75) - np.percentile(dr_boot, 25)
            # Get symmetric critical values
            cv = np.percentile(np.abs(dr_boot / se_dr_att), 95)
            # Estimate of upper boundary of 95% CI
            uci = dr_att + cv * se_dr_att
            # Estimate of lower boundary of 95% CI
            lci = dr_att - cv * se_dr_att
        else:
            # Do weighted bootstrap
            dr_boot = [
                wboot_drdid_rc(n, y, post, D, int_cov, i_weights) for _ in range(nboot)
            ]
            # Get bootstrap std errors based on IQR
            se_dr_att = np.percentile(dr_boot - dr_att, 75) - np.percentile(
                dr_boot - dr_att,
                25,
            )
            # Get symmetric critical values
            cv = np.percentile(np.abs((dr_boot - dr_att) / se_dr_att), 95)
            # Estimate of upper boundary of 95% CI
            uci = dr_att + cv * se_dr_att
            # Estimate of lower boundary of 95% CI
            lci = dr_att - cv * se_dr_att

    if not inffunc:
        dr_att_inf_func = None

    return {
        "ATT": dr_att,
        "se": se_dr_att,
        "uci": uci,
        "lci": lci,
        "boots": dr_boot,
        "att_inf_func": dr_att_inf_func,
        "call_param": None,
        "argu": {
            "panel": False,
            "estMethod": "trad",
            "boot": boot,
            "boot_type": boot_type,
            "nboot": nboot,
            "type": "dr",
        },
    }


np.random.seed(42)  # You can use any integer value as the seed


def sz_dr_dgp2():
    # Sample size
    n = 1000
    # pscore index (strength of common support)
    Xsi_ps = 0.75
    # Proportion in each period
    _lambda = 0.5
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

        # Generate treatment groups
        # Propensity score
        pi = logistic_cdf(Xsi_ps * (-x1 + 0.5 * x2 - 0.25 * x3 - 0.1 * x4))
        d = np.random.uniform(size=n) <= pi

        # Generate aux indexes for the potential outcomes
        index_lin = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)

        # Create heterogenenous effects for the ATT, which is set approximately equal to zero
        index_unobs_het = d * (index_lin)
        index_att = 0

        # This is the key for consistency of outcome regression
        index_trend = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)

        # v is the unobserved heterogeneity
        v = np.random.normal(index_unobs_het, 1)

        # Gen realized outcome at time 0
        y00 = index_lin + v + np.random.normal(size=n)
        y10 = index_lin + v + np.random.normal(size=n)

        # Gen outcomes at time 1
        # First let's generate potential outcomes: y_1_potential
        y01 = (
            index_lin + v + np.random.normal(scale=1, size=n) + index_trend
        )  # This is the baseline
        y11 = (
            index_lin + v + np.random.normal(scale=1, size=n) + index_trend + index_att
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
        dta_long["post:d"] = dta_long["post"] * dta_long["d"]
        dta_long = dta_long.sort_values(["id", "time"])

        covariates = dta_long[["x1", "x2", "x3", "x4"]].values
        y = dta_long["y"].values
        post = dta_long["post"].values
        D = dta_long["d"].values

        result = drdid_rc(y, post, D, covariates)

        ATTE_estimates.append(result["ATT"])
        asymptotic_variance.append(result["se"] ** 2)
        coverage_indicator = int(result["lci"] <= 0 <= result["uci"])
        coverage_indicators.append(coverage_indicator)

    # Calculate average bias, median bias, and RMSE
    true_ATT = 0
    average_bias = np.mean(ATTE_estimates) - true_ATT
    median_bias = np.median(ATTE_estimates) - true_ATT
    rmse = np.sqrt(np.mean((np.array(ATTE_estimates) - true_ATT) ** 2))

    # Calculate average of the variance
    average_variance = np.mean(asymptotic_variance)
    avg_coverage_prob = np.mean(coverage_indicators)
    return {
        "Average Bias": average_bias,
        "Median Bias": median_bias,
        "RMSE": rmse,
        "Average Variance of ATT": average_variance,
        "avg_coverage_prob": avg_coverage_prob,
    }
