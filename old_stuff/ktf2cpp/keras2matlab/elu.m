function y = elu(x,alpha)
    y = x;
    y(x<0) = alpha * (exp(x(x<0)) - 1) ;
end