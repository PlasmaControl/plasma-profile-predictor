function output = lstm(input,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo)
[num_inputs,input_size] = size(input);
state = zeros(2,input_size);
for i =1:num_inputs
    state = lstmcell(input(i,:),state,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo);
end
output = state(1,:);
end