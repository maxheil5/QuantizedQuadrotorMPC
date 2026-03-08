
function [epsilon, u_min_new, u_max_new, partition, mid_points] = Partition (u_min, u_max, bits)

span = u_max - u_min;

total_bins = 2^(bits); 

span_inflation = span/(2*(total_bins - 1)); % inflate the span so that after dither quantization we have no saturation; 


epsilon = (span + span_inflation)/total_bins; 

u_min_new = u_min - span_inflation/2;%keep this
u_max_new = u_max + span_inflation/2;


%%%linspace input is scalar. To do partitioning, we have to use for loop
for i=1:size(u_min,1)
    partition(i,:) = linspace( u_min_new(i,1), u_max_new(i,1), total_bins+1); 
end

mid_points = (partition(:,1:end-1) + partition(:,2:end))/2; 



end



