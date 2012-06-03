function [error_train, error_val] = ...
    random_training_validation_examples(X, y, Xval, yval, lambda)
%RANDOM_TRAINING_VALIDATION_EXAMPLES computes average training and cross validation error over i examples
%   [error_train, error_val] = ...
%       RANDOM_TRAINING_VALIDATION_EXAMPLES(X, y, Xval, yval, lambda) computes average training and cross validation error over i examples
%       for n = 1:50
%           select i random examples from training and val sets
%           compute error for each
%       endfor
%       return average error for each
%

% Number of training examples
m = size(X, 1);

error_train_m = zeros(50,1);
error_val_m = zeros(50,1);

train_num = floor(m/2);
val_num = floor(size(Xval,1)/2);
for n = 1:50
    rand_train_idx = 1 + int32(rand(train_num,1) * train_num);
    rand_val_idx = 1 + int32(rand(val_num,1) * val_num);
    rand_train_X = [ones(train_num,1), X(rand_train_idx,:)];
    rand_train_y = y(rand_train_idx,:);
    rand_val_X = [ones(val_num,1), Xval(rand_val_idx,:)];
    rand_val_y = yval(rand_val_idx,:);
    theta = trainLinearReg(rand_train_X, rand_train_y, lambda);
    [error_train_m(n), grad_train] = linearRegCostFunction(rand_train_X, rand_train_y, theta, 0);
    [error_val_m(n), grad_val] = linearRegCostFunction(rand_val_X, rand_val_y, theta, 0);
endfor

error_train = mean(error_train_m);
error_val = mean(error_val_m);
end
