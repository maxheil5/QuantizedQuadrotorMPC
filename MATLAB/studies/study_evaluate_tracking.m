function tracking_result = study_evaluate_tracking(context, EDMD_model, reference, capture_trace)
if nargin < 4
    capture_trace = false;
end

mpc_result = sim_MPC(EDMD_model, context.Z0, reference.Z_ref, reference.X_ref, context.params);

tracking_result = struct();
tracking_result.reference_name = reference.name;
tracking_result.reference_label = reference.label;
tracking_result.e_MPC = study_position_rmse(mpc_result.X(:, 1:3), mpc_result.X_ref(:, 1:3));

if capture_trace
    tracking_result.mpc = mpc_result;
else
    tracking_result.mpc = [];
end
end
