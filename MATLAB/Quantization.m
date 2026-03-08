function [u_quantized] = Quantization(u_min, u_max, u, bin_length, mid_points) 


 idx =  ceil((u-u_min)/bin_length); 


 if u >= u_max
     u_quantized = mid_points(end);
     fprintf('\n Saturation happened'); % saturation should not happen
 
 elseif u <= u_min
     u_quantized = mid_points(1);
     fprintf('\n Saturation happened'); % saturation should not happen
 
 else
     u_quantized = mid_points(idx);
 end



end

