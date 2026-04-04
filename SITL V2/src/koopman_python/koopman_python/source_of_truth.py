"""V2 source-of-truth mapping.

This package is intentionally fresh. The current ROS 2 SITL implementation is
not imported here.
"""

MATLAB_SOURCES = {
    "dynamics_params": "MATLAB/dynamics/get_params.m",
    "dynamics_srb": "MATLAB/dynamics/dynamics_SRB.m",
    "training_random": "MATLAB/training/get_rnd_trajectories.m",
    "quantized_main": "MATLAB/main_Heil_FINAL_V2.m",
    "paper_main": "MATLAB/main.m",
}

KOOPMAN_REFERENCE = {
    "repo": "SITL V2/src/KoopmanMPC_Quadrotor",
    "basis": "edmd/get_basis.m",
    "fit": "edmd/get_EDMD.m",
    "qp": "mpc/get_QP.m",
    "simulate": "mpc/sim_MPC.m",
}

