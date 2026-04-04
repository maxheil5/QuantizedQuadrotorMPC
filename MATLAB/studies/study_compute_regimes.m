function [regime_labels, near_start_bit] = study_compute_regimes(bits, median_e_pred, median_e_mpc)
bits = bits(:);
median_e_pred = median_e_pred(:);
median_e_mpc = median_e_mpc(:);

reference_idx = find(bits == max(bits), 1, 'first');
reference_pred = median_e_pred(reference_idx);
reference_mpc = median_e_mpc(reference_idx);

near_start_bit = bits(end);
for idx = 1:numel(bits)
    higher_mask = idx:numel(bits);
    pred_ok = all(median_e_pred(higher_mask) <= 1.10 * reference_pred);
    mpc_ok = all(median_e_mpc(higher_mask) <= 1.10 * reference_mpc);
    if pred_ok && mpc_ok
        near_start_bit = bits(idx);
        break
    end
end

near_idx = find(bits == near_start_bit, 1, 'first');
near_pred = median_e_pred(near_idx);
near_mpc = median_e_mpc(near_idx);

regime_labels = strings(numel(bits), 1);
for idx = 1:numel(bits)
    if bits(idx) >= near_start_bit
        regime_labels(idx) = "near-saturation";
    elseif median_e_pred(idx) >= 2 * near_pred || median_e_mpc(idx) >= 2 * near_mpc
        regime_labels(idx) = "coarse";
    else
        regime_labels(idx) = "transition";
    end
end
end
