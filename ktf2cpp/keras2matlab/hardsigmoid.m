function y = hardsigmoid(x)
y=.2*x+.5;
y(x<-2.5) = 0;
y(x>2.5) = 1;
end