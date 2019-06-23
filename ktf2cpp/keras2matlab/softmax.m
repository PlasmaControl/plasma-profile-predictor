function y = softmax(x)
    y = exp(x-max(x));
    y = y ./sum(y);
end