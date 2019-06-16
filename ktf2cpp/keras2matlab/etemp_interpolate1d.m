function y_interp = etemp_interpolate1d(x,y,x_interp)
assert(length(x)<100);
assert(length(y)<100);
assert(length(x_interp)<100);

y_interp = interp1(x,y,x_interp,'linear','extrap');
end

