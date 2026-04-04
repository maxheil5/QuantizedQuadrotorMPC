function model_result = study_identify_model(context, bits, mode, trial_seed)
model_result = struct();
model_result.bits = bits;
model_result.mode = string(mode);
model_result.trial_seed = trial_seed;

if string(mode) == "none" || ~isfinite(bits)
    EDMD_model = context.EDMD;
    e_A = 0;
    e_B = 0;
else
    if string(mode) == "dithered"
        rng(trial_seed, 'twister');
    end
    quantized = study_quantize_training_data(context, bits, mode);
    EDMD_model = get_EDMD(quantized.X1, quantized.X2, quantized.U1, context.n_basis, context.t_traj);
    EDMD_model.n_basis = context.n_basis;
    e_A = norm(context.EDMD.A - EDMD_model.A, 'fro') / norm(context.EDMD.A, 'fro');
    e_B = norm(context.EDMD.B - EDMD_model.B, 'fro') / norm(context.EDMD.B, 'fro');
end

EDMD_model.n_basis = context.n_basis;
evalc('[RMSE_eval, ~] = eval_EDMD_fixed_traj(context.X0, context.X_eval, context.U_eval, context.dt, EDMD_model, context.n_basis);');

model_result.EDMD = EDMD_model;
model_result.e_A = e_A;
model_result.e_B = e_B;
model_result.e_pred = RMSE_eval.x;
end
