function dithered_signal = Dither_Func(Input_data, epsilon, min_new, max_new, partition, midPoints)


%Generating Noise to be added to the signal
% Dither_Signal = epsilon.*(rand(size(Input_data))-0.5);

Dither_Signal = (epsilon/2).*(rand(size(Input_data))-0.5);
%Adding Noise to the data
Input_data_noiseAdded = Input_data + Dither_Signal;
%Quantizing Noisy Data
Input_data_noiseAdded_quantized = quantizeCustom(min_new, max_new, Input_data_noiseAdded, epsilon, midPoints);
%Subtracting noise from quantized noisy data
Input_data_noiseAdded_quantized_noiseSubtracted = Input_data_noiseAdded_quantized -Dither_Signal;
%returning the dithered version of input
dithered_signal = Input_data_noiseAdded_quantized_noiseSubtracted;


end