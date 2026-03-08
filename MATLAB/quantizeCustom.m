function [quantized_data] = quantizeCustom(min, max, data, binLength, midPoints)

quantized_data = zeros(size(data));


for i = 1:size(data,1)
    for j=1:size(data,2)
        quantized_data(i,j) = Quantization(min(i,:), max(i,:), data(i,j), binLength(i,:), midPoints(i,:));
    end
end

end