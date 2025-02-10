function [loss, acc] = compute_loss(X, Y, W1, b1, W2, b2)
    [Y_hat, ~] = forward_pass(X, W1, b1, W2, b2);
    N = size(X,2);
    loss = -sum(sum(Y.*log(Y_hat+1e-12)))/N;
    [~, pred] = max(Y_hat, [], 1);
    [~, labels] = max(Y, [], 1);
    acc = mean(pred == labels);
end