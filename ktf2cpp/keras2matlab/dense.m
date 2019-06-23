function y = dense(x,A,b)
y = zeros(size(A,2));
y = x*A+b;
end