
te_raw_rho = linspace(0,1,10);
te_raw = linspace(0,2,10);
pinj = 4;
ipp = 4.5;
filtered_signals = eTpred_prep_inputs(pinj,ipp,te_raw,te_raw_rho);