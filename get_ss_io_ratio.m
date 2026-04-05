function result = get_ss_io_ratio(u, y, varargin)
%GET_SS_IO_RATIO Compute steady-state input/output ratio from trajectories.
%
% result = get_ss_io_ratio(u, y)
% result = get_ss_io_ratio(u, y, 'Name', Value, ...)
%
% Inputs
%   u : [N x nu] input trajectory
%   y : [N x ny] output trajectory
%
% Name-Value Options
%   'WindowLength' : number of samples used for steady-state averaging
%                    default = max(10, round(0.1*N))
%   'UseAbsInput'  : if true, ratio is y_ss ./ abs(u_ss), default = false
%   'EpsDen'       : small threshold to avoid division by zero, default = 1e-9
%
% Outputs
%   result.u_ss        : steady-state mean of input
%   result.y_ss        : steady-state mean of output
%   result.ratio       : steady-state ratio y_ss ./ u_ss
%   result.idxWindow   : index range used as steady-state window
%
% Notes
%   1) This function assumes the last part of the trajectory is near steady state.
%   2) For MIMO, ratio is returned elementwise only when ny == nu.
%      If ny ~= nu, only u_ss and y_ss are returned, and ratio is set to [].

    % -------------------------
    % Input check
    % -------------------------
    if nargin < 2
        error('At least u and y must be provided.');
    end

    validateattributes(u, {'numeric'}, {'2d', 'nonempty', 'finite'}, mfilename, 'u', 1);
    validateattributes(y, {'numeric'}, {'2d', 'nonempty', 'finite'}, mfilename, 'y', 2);

    Nu = size(u, 1);
    Ny = size(y, 1);

    if Nu ~= Ny
        error('u and y must have the same number of rows (time samples).');
    end

    N = Nu;

    % -------------------------
    % Parse options
    % -------------------------
    p = inputParser;
    p.addParameter('WindowLength', max(10, round(0.1 * N)), ...
        @(x) isnumeric(x) && isscalar(x) && (x >= 1));
    p.addParameter('UseAbsInput', false, ...
        @(x) islogical(x) && isscalar(x));
    p.addParameter('EpsDen', 1e-9, ...
        @(x) isnumeric(x) && isscalar(x) && (x > 0));
    p.parse(varargin{:});

    winLen      = min(N, round(p.Results.WindowLength));
    useAbsInput = p.Results.UseAbsInput;
    epsDen      = p.Results.EpsDen;

    idx1 = N - winLen + 1;
    idx2 = N;

    u_win = u(idx1:idx2, :);
    y_win = y(idx1:idx2, :);

    % -------------------------
    % Steady-state mean values
    % -------------------------
    u_ss = mean(u_win, 1);
    y_ss = mean(y_win, 1);

    % -------------------------
    % Ratio
    % -------------------------
    ratio = [];

    if size(u, 2) == size(y, 2)
        den = u_ss;
        if useAbsInput
            den = abs(den);
        end

        ratio = nan(size(y_ss));
        valid = abs(den) > epsDen;
        ratio(valid) = y_ss(valid) ./ den(valid);
    else
        warning(['For MIMO with different input/output dimensions, elementwise ratio is not defined. ' ...
                 'Only u_ss and y_ss are returned.']);
    end

    % -------------------------
    % Output
    % -------------------------
    result = struct();
    result.u_ss      = u_ss;
    result.y_ss      = y_ss;
    result.ratio     = ratio;
    result.idxWindow = idx1:idx2;
end