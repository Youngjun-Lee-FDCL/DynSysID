function [yMinOut, yMaxOut] = compute_nice_ylim(y)

yMin = min(y(:));
yMax = max(y(:));

range = yMax - yMin;

if range < 1e-8
    % flat signal 대응
    margin = max(abs(yMax)*0.1, 1e-3);
else
    margin = 0.05 * range;   % 5% padding
end

yMinOut = yMin - margin;
yMaxOut = yMax + margin;

end