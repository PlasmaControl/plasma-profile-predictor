function y = dense(x,A,b,relu)
y = zeros(size(x));
y = x*A+b;
if relu
    y = max(y,0);
end

end