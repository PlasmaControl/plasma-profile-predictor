#!/usr/bin/env python3


import numpy as np
import keras
import argparse
import sys, os



def parse_args(args):
    """Parses command line arguments

    Args:
        args (list): List of arguments to parse, ie sys.argv

    Returns:
        parsed_args (argparse object): argparse object containing parsed
        arguments.
    """

    parser = argparse.ArgumentParser(prog='keras2matlab',
                                     description="""This script takes a keras model (.h5) and generates matlab code to mimic it""")
    parser.add_argument("model_path", help="file path to keras .h5 model file")
    parser.add_argument("function_name", help="matlab function name to use")

    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    """Runs the main script, from the command line.

    """

    args = parse_args(args)

    keras2matlab(args.model_path,args.function_name)


    

def array2matlab(array,name):
    s = name + ' = single(['
    shp = array.shape
    count=0
    if len(shp) is 1:
        for i in range(shp[0]):
            s += "{:.16f}".format(array[i]) + ','
            count += 1
            if (count)%5 is 0:
                s += '...\n'
                
    elif len(shp) is 2:
        for i in range(shp[0]):
            for j in range(shp[1]):
                s += "{:.16f}".format(array[i,j])
                count += 1
                if count%shp[1] is 0:
                    s += ';'
                else:
                    s += ','
                if (count)%5 is 0:
                    s += '...\n'
            
    s = s + ']); \n'
    return s


def weights2matlab(model,file):
    for layer in model.layers[1:]:
        weights = layer.get_weights()
        if 'dense' in layer.name:
            A = weights[0]
            b = weights[1]
            if len(layer.output_shape) is 3:
                b= np.tile(b,(layer.output_shape[1],1))
            file.write(array2matlab(A,layer.name + 'A'))
            file.write(array2matlab(b,layer.name + 'b'))
        elif 'lstm' in layer.name:
            W = weights[0]
            U = weights[1]
            b = weights[2]
            units = layer.get_config()['units']

            Wi = W[:, :units]
            Wf = W[:, units: units * 2]
            Wc = W[:, units * 2: units * 3]
            Wo = W[:, units * 3:]

            Ui = U[:, :units]
            Uf = U[:, units: units * 2]
            Uc = U[:, units * 2: units * 3]
            Uo = U[:, units * 3:]

            bi = b[:units]
            bf = b[units: units * 2]
            bc = b[units * 2: units * 3]
            bo = b[units * 3:]
            
            file.write(array2matlab(Wi,layer.name + 'Wi'))
            file.write(array2matlab(Wf,layer.name + 'Wf'))
            file.write(array2matlab(Wc,layer.name + 'Wc'))
            file.write(array2matlab(Wo,layer.name + 'Wo'))

            file.write(array2matlab(Ui,layer.name + 'Ui'))
            file.write(array2matlab(Uf,layer.name + 'Uf'))
            file.write(array2matlab(Uc,layer.name + 'Uc'))
            file.write(array2matlab(Uo,layer.name + 'Uo'))

            file.write(array2matlab(bi,layer.name + 'bi'))
            file.write(array2matlab(bf,layer.name + 'bf'))
            file.write(array2matlab(bc,layer.name + 'bc'))
            file.write(array2matlab(bo,layer.name + 'bo'))
    return


def model2matlab(model,file,name):
    s = 'function prediction = ' + name + '(input)'
    file.write(s + '\n \n')
    weights2matlab(model,file)
    count = 1
    activations_to_write = []
    layers_to_write = []
    for layer in model.layers[1:]:
        if len(layer.output_shape[1:]) is 2:
            s = 'layer'+str(count) + 'out = single(zeros' + str(layer.output_shape[1:]) + ');'
            file.write(s + '\n')
        elif len(layer.output_shape[1:]) is 1:
            s = 'layer'+str(count) + 'out = single(zeros(1,' + str(layer.output_shape[1]) + '));'
            file.write(s + '\n')
        count += 1
    file.write('layer0out = input; \n \n')
    count = 1
    for layer in model.layers[1:]:
        if 'dense' in layer.name:
            layers_to_write.append(layer)
            s = 'layer' + str(count) + 'out = dense(layer' + str(count-1) + 'out,' + layer.name + 'A,' + layer.name + 'b);'
            file.write(s + '\n')
            if 'linear' not in layer.get_config()['activation']:
                if layer.get_config()['activation'] not in activations_to_write:
                    activations_to_write.append(layer.get_config()['activation'])
                s = 'layer' + str(count) + 'out = ' + layer.get_config()['activation'] + '(layer' + str(count) + 'out);'
                file.write(s + '\n')
        if 'lstm' in layer.name:
            layers_to_write.append(layer)
            if layer.get_config()['activation'] not in activations_to_write:
                    activations_to_write.append(layer.get_config()['activation'])
            if layer.get_config()['recurrent_activation'] not in activations_to_write:
                    activations_to_write.append(layer.get_config()['recurrent_activation'])
            s_weights = ['Wi','Wf','Wc','Wo','Ui','Uf','Uc','Uo','bi','bf','bc','bo']
            s_weights = [layer.name + suf for suf in s_weights]
            s_weights = ",".join(s_weights)
            s = 'layer' + str(count) + 'out = ' + layer.name + '(layer' + str(count-1) + 'out,' + s_weights + ');'
            file.write(s + '\n')
        count += 1
    file.write('prediction = layer' + str(len(model.layers[1:])) + 'out; \n end') 
    return layers_to_write, activations_to_write



def layers2matlab(layers):
    densewritten = False
    for layer in layers:
        if ('dense' in layer.name) and not densewritten:
            densewritten = True
            file = open('dense.m',"w+")
            s = """function y = dense(x,A,b)
y = zeros(size(A,2));
y = x*A+b;
end"""
            file.write(s)
            file.close()
        if 'lstm' in layer.name:
            file = open(layer.name + '.m',"w+")
            s = 'function output = ' + layer.name + '(input,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo) \n'
            s += '[num_inputs,input_size] = size(input); \n'
            s += 'state = single(zeros(2,input_size)); \n'
            s += 'for i =1:num_inputs \n'
            s += 'state = ' + layer.name + 'cell(input(i,:),state,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo); \n'
            s += 'end \n'
            s += 'output = state(1,:); \n'
            s += 'end'
            file.write(s)
            file.close()
            
            file = open(layer.name + 'cell.m',"w+")
            s = 'function state = ' + layer.name + 'cell(inputs,states,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo) \n'
            s += """
h_tm1 = states(1,:);  % previous memory state
c_tm1 = states(2,:);  % previous carry state

inputs_i = inputs;
inputs_f = inputs;
inputs_c = inputs;
inputs_o = inputs;

x_i = inputs_i*Wi;
x_f = inputs_f*Wf;
x_c = inputs_c*Wc;
x_o = inputs_o*Wo;

x_i = x_i +bi;
x_f = x_f +bf;
x_c = x_c +bc;
x_o = x_o +bo;

h_tm1_i = h_tm1;
h_tm1_f = h_tm1;
h_tm1_c = h_tm1;
h_tm1_o = h_tm1;

"""
            s += 'yi = ' + layer.get_config()['recurrent_activation'] + '(x_i + h_tm1_i*Ui); \n' 
            s += 'yf = ' + layer.get_config()['recurrent_activation'] + '(x_f + h_tm1_f*Uf); \n'
            s += 'yc = yf.*c_tm1 + yi .* ' + layer.get_config()['activation'] + '(x_c + h_tm1_c*Uc); \n'
            s += 'yo = ' + layer.get_config()['recurrent_activation'] + '(x_o + h_tm1_o*Uo); \n'
            s += 'h = yo .* ' + layer.get_config()['activation'] + '(yc); \n'
            s += """state = [h;yc];

end
"""
            file.write(s)
            file.close()
    return


def activations2matlab(activations):
    if 'hard_sigmoid' in activations:
        file = open('hard_sigmoid.m',"w+")
        s = """function y = hard_sigmoid(x)
y=.2*x+.5;
y(x<-2.5) = 0;
y(x>2.5) = 1;
end"""
        file.write(s)
        file.close()
    if 'relu' in activations:
        file = open('relu.m',"w+")
        s = """function y = relu(x)
y=x;
y(y<0) = 0;
end"""
        file.write(s)
        file.close()
    if 'softmax' in activations:
        file = open('softmax.m',"w+")
        s = """function y = softmax(x)
y = exp(x-max(x));
y = y ./sum(y);
end"""
        file.write(s)
        file.close()
    if 'softplus' in activations:
        file = open('softplus.m',"w+")
        s = """function y = softplus(x)
y = log(1+exp(x));
end"""
        file.write(s)
        file.close()
    if 'softsign' in activations:
        file = open('softsign.m',"w+")
        s = """function y = softsign(x)
y = x./(1+abs(x));
end"""
        file.write(s)
        file.close()
    if 'elu' in activations:
        file = open('elu.m',"w+")
        s = """function y = elu(x,alpha)
y = x;
y(x<0) = alpha * (exp(x(x<0)) - 1) ;
end"""
        file.write(s)
        file.close()
    if 'sigmoid' in activations:
        file = open('sigmoid.m',"w+")
        s = """function y = sigmoid(x)
y = 1 ./ (1+exp(-x));
end"""
        file.write(s)
        file.close()
    return

def make_test_suite(model,function_name,num_tests=10):
    file = open(function_name + '_test_suite.m',"w+")
    for i in range(num_tests):
        rand_input = np.random.random(model.layers[0].input_shape[1:])
        file.write(array2matlab(rand_input,'rand_input' + str(i+1)))
        rand_input = rand_input[np.newaxis,...]
        output = model.predict(rand_input)
        file.write(array2matlab(output,'keras_output' + str(i+1)))
    for i in range(num_tests):
        s = 'matlab_output' + str(i+1) + ' = ' + function_name + '(rand_input' + str(i+1) + '); \n'
        file.write(s)
        s = 'error(' + str(i+1) + ') = norm(keras_output' + str(i+1) + '-matlab_output' + str(i+1) + '); \n'
        file.write(s)
    s = 'error \n'
    file.write(s)
    file.close()
    return


def keras2matlab(model_filepath,function_name):
    print('Loading model file')
    model = keras.models.load_model(model_filepath)
    dirName = function_name + '_matlab_code'
 
    try:
        # Create target Directory
        os.mkdir(dirName)
        os.chdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        return
    matlab_filepath = './' + function_name + '.m'
    file = open(matlab_filepath,"w+")
    print('Writing weights and architecture')
    (layers_to_write, activations_to_write) = model2matlab(model,file,function_name)
    file.close()
    print('Writing activations and layers')
    activations2matlab(activations_to_write)
    layers2matlab(layers_to_write)
    make_test_suite(model,function_name)
    print('Done, all files are in directory: ' + os.getcwd())
    os.chdir('..')



if __name__ == '__main__':
    main(sys.argv[1:])
