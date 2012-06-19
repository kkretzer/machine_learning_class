%% Initialization
clear ; close all; clc

X = [];
y = [];

sample_size = 1;
fprintf('\nGetting spam assassin public corpus "ham" file listing...\n');
fflush(stdout);
ham_files = glob('spam_assassin_public_corpus/*ham*/[0-9]*');
% cut this down to sample_size
for i = 1:(size(ham_files,1) - sample_size)
    ham_files(1 + int32(rand(1,1)*(size(ham_files,1)-1))) = []; % delete random element
endfor

fprintf('\nGetting spam assassin public corpus "spam" file listing...\n');
fflush(stdout);
spam_files = glob('spam_assassin_public_corpus/*spam*/[0-9]*');
% cut this down to sample_size
for i = 1:(size(spam_files,1) - sample_size)
    spam_files(1 + int32(rand(1,1)*(size(spam_files,1)-1))) = []; % delete random element
endfor

fprintf('\nReading, processing, and extracting features from ham files...\n');
fflush(stdout);
for i = 1:size(ham_files, 1)
    X(end+1) = emailFeatures(processEmail(readFile(ham_files{i})))';
    y(end+1) = 0;
endfor

fprintf('\nReading, processing, and extracting features from spam files...\n');
fflush(stdout);
for i = 1:size(spam_files, 1)
    X(end+1) = emailFeatures(processEmail(readFile(spam_files{i})))';
    y(end+1) = 1;
endfor

X_train = [];
y_train = [];
X_val = [];
y_val = [];
X_test = [];
y_test = [];

fprintf('\nBreaking up data into training, cross-validation, and test sets...\n');
fflush(stdout);
for i = 1:size(X,1)
    r = rand(1,1);
    if (r <= 0.8)
        X_train(end+1) = X(i);
        y_train(end+1) = y(i);
    elseif (r <= 0.9)
        X_val(end+1) = X(i);
        y_val(end+1) = y(i);
    else
        X_test(end+1) = X(i);
        y_test(end+1) = y(i);
    endif
endfor

fprintf('\n Training set size = %d\n', size(y_train, 1));
fprintf('\n Cross-validation set size = %d\n', size(y_val, 1));
fprintf('\n Test set size = %d\n', size(y_test, 1));
fflush(stdout);

possible_C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100];
C_val_error = zero(size(possible_C,1), 1);
models = zero(size(possible_C,1), 1);

for i = i:size(possible_C, 1)
    C = possible_C(i);
    model = svmTrain(X_train, y_train, C, @linearKernel);
    models(end+1) = model;
    
    fprintf('\nTraining with C=%g\n', C);
    fflush(stdout);
    p = svmPredict(model, X_val);
    C_val_error(i) = mean(double(p == y_val)) * 100;
    fprintf('  Cross-validation Accuracy: %f\n', C_val_error(i));
    fflush(stdout);
endfor

[min_C_val_error, min_C_val_error_idx] = min(C_val_error);
fprintf('\nOptimal value of C=%g with cross-validation error of %g\n', possible_C(min_C_val_error_idx), C_val_error(min_C_val_error_idx));

fprintf('\nRetrieving the final model...\n');
fflush(stdout);
model = models(min_C_val_error);
fprintf('\nTesting the final model...\n');
fflush(stdout);
p = svmPredict(model, X_test);

fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);
