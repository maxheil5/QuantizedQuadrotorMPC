function context = study_build_context(opts)
rng(opts.rng_seed, 'twister');
warning('off', 'optim:quadprog:HessianNotSym');

params = get_params();
params.use_casadi = false;
params.predHorizon = 10;
params.simTimeStep = 1e-3;
params.SimTimeDuration = 1.2;
params.MAX_ITER = floor(params.SimTimeDuration / params.simTimeStep);
params.show_waitbar = opts.show_waitbar;
params.verbose = false;

context = struct();
context.dt = 1e-3;
context.t_span = 0.1;
context.t_traj = 0:context.dt:context.t_span;
context.n_control = 100;
context.n_basis = 3;
context.params = params;
context.training_flag = 'train';

x0 = [0;0;0];
dx0 = [0;0;0];
R0 = eye(3);
wb0 = [0.1;0;0];
context.X0 = [x0; dx0; R0(:); wb0];

[context.X_train, context.U_train, context.X1_train, context.X2_train, context.U1_train] = ...
    get_rnd_trajectories(context.X0, context.n_control, context.t_traj, false, context.training_flag);

context.EDMD = get_EDMD(context.X1_train, context.X2_train, context.U1_train, context.n_basis, context.t_traj);
context.EDMD.n_basis = context.n_basis;

[context.X_eval, context.U_eval] = get_rnd_trajectories(context.X0, 1, context.t_traj, false, context.training_flag);

basis0 = get_basis(context.X0, context.n_basis);
context.Z0 = [context.X0(1:3); context.X0(4:6); basis0];
context.reference_horizon = context.params.MAX_ITER + context.params.predHorizon - 1;

t_ref = 0:context.params.simTimeStep:10;
X_random = get_rnd_trajectories(context.X0, 1, t_ref, false, 'mpc');
X_random = X_random(:, 2:end);
context.default_reference = study_build_reference_struct('random_reference', 'Random reference', X_random, context);
context.reference_library = study_build_reference_library(context);
end
