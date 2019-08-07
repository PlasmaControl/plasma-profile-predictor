lstm0 = dlmread('lstm0.txt');
lstm1 = dlmread('lstm1.txt');
lstm2 = dlmread('lstm2.txt');
dense1A = dlmread('dense1A.txt');
dense1B = dlmread('dense1B.txt')';
dense2A = dlmread('dense2A.txt');
dense2B = dlmread('dense2B.txt')';
dense3A = dlmread('dense3A.txt');
dense3B = dlmread('dense3B.txt')';
units = 20;
Wi = lstm0(:, 1:units);
Wf = lstm0(:, units+1: units * 2);
Wc = lstm0(:, units * 2+1: units * 3);
Wo = lstm0(:, units * 3+1:end);

Ui = lstm1(:, 1:units);
Uf = lstm1(:, units+1: units * 2);
Uc = lstm1(:, units * 2+1: units * 3);
Uo = lstm1(:, units * 3+1:end);

bi = lstm2(1:units)';
bf = lstm2(units+1: units * 2)';
bc = lstm2(units * 2+1: units * 3)';
bo = lstm2(units * 3+1:end)';


input = linspace(0,10,8*32);
input = single(reshape(input,[8,32]));


prediction = profile_predictor(input);
% plot(prediction)