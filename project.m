% load training and test data
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% convert labels to one-hot encoding:
numClasses = 10;
Y_train = zeros(numClasses, length(train_labels));
for i = 1:length(train_labels)
    Y_train(train_labels(i)+1, i) = 1;
end
Y_test = zeros(numClasses, length(test_labels));
for i = 1:length(test_labels)
    Y_test(test_labels(i)+1, i) = 1;
end

X_train = train_images; % 784 * 60000
X_test = test_images; % 784 * 10000

% create a validation set from training data to verify the correctness on 
% training sets to have a basic idea of accuracy.
val_size = 5000;
X_val = X_train(:, end-val_size+1:end);
Y_val = Y_train(:, end-val_size+1:end);
X_train = X_train(:, 1:end-val_size);
Y_train = Y_train(:, 1:end-val_size);

% initialize Neural Network Parameters
% We have 28x28 = 784
inputSize = 784;
hiddenSize = 100;
outputSize = 10;

% make sure the random numbers generated are the same every time
rng('default');
% W1 is a matrix of size hiddenSize * inputSize
% initialize some small random values and scaled by 0.01
W1 = 0.01*randn(hiddenSize, inputSize);
b1 = zeros(hiddenSize, 1);
W2 = 0.01*randn(outputSize, hiddenSize);
b2 = zeros(outputSize, 1);

% Mini-Batch Gradient Descent Training
numEpochs = 5;
batchSize = 128;
% change the learning_rate
learning_rate = 0.87;
fprintf('Learning Rate is %f \n', learning_rate);
numTrain = size(X_train, 2);

for epoch = 1:numEpochs
    % shuffle training data
    idx = randperm(numTrain);
    X_train = X_train(:, idx);
    Y_train = Y_train(:, idx);
    
    % loop over mini-batches
    for start_i = 1:batchSize:numTrain
        end_i = min(start_i+batchSize-1, numTrain);
        X_batch = X_train(:, start_i:end_i);
        Y_batch = Y_train(:, start_i:end_i);
        [loss, dW1, db1, dW2, db2] = backward_pass(X_batch, Y_batch, W1, b1, W2, b2);
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
    end
    
    % compute loss on training and validation set
    [train_loss, ~] = compute_loss(X_train, Y_train, W1, b1, W2, b2);
    [val_loss, val_acc] = compute_loss(X_val, Y_val, W1, b1, W2, b2);
    fprintf('Epoch %d: Train Loss = %.4f | Val Loss = %.4f | Val Acc = %.2f%%\n', ...
        epoch, train_loss, val_loss, val_acc*100);
end

% evaluate on test sets
[test_loss, test_acc] = compute_loss(X_test, Y_test, W1, b1, W2, b2);
fprintf('Test Loss = %.4f | Test Accuracy = %.2f%%\n', test_loss, test_acc*100);

% test set confusion matrix
[Y_hat_test, ~] = forward_pass(X_test, W1, b1, W2, b2);
[~, pred_test] = max(Y_hat_test, [], 1);
[~, true_test] = max(Y_test, [], 1);
confMat = confusionmat(true_test, pred_test);
disp('Confusion Matrix:');
disp(confMat);

% normalize the confusion matrix
confMatNorm = bsxfun(@rdivide, confMat, sum(confMat,2));
imagesc(confMatNorm);
colorbar;
title('Normalized Confusion Matrix');
xlabel('Predicted Digit');
ylabel('True Digit');
axis square;
