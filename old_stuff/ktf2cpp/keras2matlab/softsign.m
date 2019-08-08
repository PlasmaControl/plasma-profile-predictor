function y = softsign(x)
    y = x./(1+abs(x));
end