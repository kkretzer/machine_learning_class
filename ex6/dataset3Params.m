function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30](:);
range_size = size(range, 1);
prediction_error = zeros(range_size, range_size);

for try_C = 1:range_size
    for try_sigma = 1:range_size
        model = svmTrain(X, y, range(try_C, 1), @(x1, x2) gaussianKernel(x1, x2, range(try_sigma, 1)));
        predictions = svmPredict(model, Xval);
        prediction_error(try_C, try_sigma) = mean(double(predictions ~= yval));
    endfor
endfor

[col_mins, col_mins_idx] = min(prediction_error);
[min_min, sigma_min_idx] = min(col_mins);
C_min_idx = col_mins_idx(sigma_min_idx);
C = range(C_min_idx);
sigma = range(sigma_min_idx);

% =========================================================================

end
