"""Code for the core analyses."""
from deep_learning_for_semiparametric_did_estimation.analysis.dr_did import (
    estimate_dr_ATTE,
)
from deep_learning_for_semiparametric_did_estimation.analysis.ipw_dgp1 import (
    ipw_sim_dgp1,
)
from deep_learning_for_semiparametric_did_estimation.analysis.ipw_dgp2 import (
    ipw_sim_dgp2,
)
from deep_learning_for_semiparametric_did_estimation.analysis.ipw_dgp3 import (
    ipw_sim_dgp3,
)
from deep_learning_for_semiparametric_did_estimation.analysis.ipw_dgp4 import (
    ipw_sim_dgp4,
)
from deep_learning_for_semiparametric_did_estimation.analysis.twfe_dgp1 import (
    twfe_DGP1_simulation,
)
from deep_learning_for_semiparametric_did_estimation.analysis.twfe_dgp2 import (
    twfe_DGP2_simulation,
)
from deep_learning_for_semiparametric_did_estimation.analysis.twfe_dgp3 import (
    twfe_DGP3_simulation,
)
from deep_learning_for_semiparametric_did_estimation.analysis.twfe_dgp4 import (
    twfe_DGP4_simulation,
)

__all__ = [
    estimate_dr_ATTE,
    twfe_DGP1_simulation,
    twfe_DGP2_simulation,
    twfe_DGP3_simulation,
    twfe_DGP4_simulation,
    ipw_sim_dgp1,
    ipw_sim_dgp2,
    ipw_sim_dgp3,
    ipw_sim_dgp4,
]
