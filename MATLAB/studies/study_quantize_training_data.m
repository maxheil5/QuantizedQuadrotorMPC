function quantized = study_quantize_training_data(context, bits, mode)
quantized = struct();

switch string(mode)
    case "none"
        quantized.X = context.X_train;
        quantized.X1 = context.X1_train;
        quantized.X2 = context.X2_train;
        quantized.U1 = context.U1_train;
    case {"dithered", "standard"}
        X_min = min(context.X_train, [], 2);
        X_max = max(context.X_train, [], 2);
        [epsilon_X, X_min_new, X_max_new, partition_X, midPoints_X] = Partition(X_min, X_max, bits);

        U_min = min(context.U1_train, [], 2);
        U_max = max(context.U1_train, [], 2);
        [epsilon_U, U_min_new, U_max_new, partition_U, midPoints_U] = Partition(U_min, U_max, bits);

        if string(mode) == "dithered"
            quantized.X = Dither_Func(context.X_train, epsilon_X, X_min_new, X_max_new, partition_X, midPoints_X);
            quantized.U1 = Dither_Func(context.U1_train, epsilon_U, U_min_new, U_max_new, partition_U, midPoints_U);
        else
            quantized.X = quantizeCustom(X_min_new, X_max_new, context.X_train, epsilon_X, midPoints_X);
            quantized.U1 = quantizeCustom(U_min_new, U_max_new, context.U1_train, epsilon_U, midPoints_U);
        end

        [quantized.X1, quantized.X2] = study_split_state_snapshots(quantized.X, context.n_control, numel(context.t_traj));
    otherwise
        error('Unsupported quantization mode: %s', mode);
end
end
