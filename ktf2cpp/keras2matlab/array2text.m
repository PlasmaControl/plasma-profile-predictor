function text = array2text(array)
text = '[';
[row,col] = size(array);
counter = 0;
for i = 1:row
    for j = 1:col
        text = strcat(text,num2str(array(i,j),16));
        counter = counter+1;
        if mod(counter,col) ==0
            text = strcat(text, ';');
        else
            text = strcat(text, ',');
        end
        if mod(counter,5)==0
            text = strcat(text,'... \n');
        end
    end
end
text = strcat(text, '];');
text = sprintf(text);
end