function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = (theta' * X')'

% don't regularize theta(1)
theta_reg = theta(2 : size(theta,1))

cost_regularization = lambda * sum(theta_reg .^ 2) / (2 * m)
J = sum(-y .* log(sigmoid(z)) - (1 -y) .* log(1-sigmoid(z))) / m + cost_regularization

grad_regularization =  lambda * [0; theta_reg] / m
grad = ((sigmoid(z) - y)' * X)' / m + grad_regularization


% =============================================================

end
