function plotDataPoints(X, idx, K)
%PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
%index assignments in idx have the same color
%   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
%   with the same index assignments in idx have the same color

% original provided code:
% % Create palette
% palette = hsv(K + 1);
% colors = palette(idx, :);
% 
% % Plot the data
% scatter(X(:,1), X(:,2), 15, colors);
% END original provided code:

% this works around an octave bug, i think with mac osx
colormap(hsv(K + 1)(1:end-1, :));
scatter(X(:, 1), X(:, 2), 15, idx);

end
