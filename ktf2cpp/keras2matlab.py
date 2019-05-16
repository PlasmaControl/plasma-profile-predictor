#!/usr/bin/env python3


import numpy as np
import keras
import argparse
import sys



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
    s = name + ' = ['
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
            
    s = s + ']; \n'
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


def model2matlab(model,file,name):
    s = 'function prediction = ' + name + '(input)'
    file.write(s + '\n \n')
    print('Writing model weights \n')
    weights2matlab(model,file)
    print('Writing model architecture \n')
    count = 1
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
            s = 'layer' + str(count) + 'out = layer' + str(count-1) + 'out*' + layer.name + 'A+' + layer.name + 'b;'
            file.write(s + '\n')
            if 'relu' in layer.get_config()['activation']:
                s = 'layer' + str(count) + 'out(layer' + str(count) + 'out<0) = 0;'
                file.write(s + '\n')
        if 'lstm' in layer.name:
            s_weights = ['Wi','Wf','Wc','Wo','Ui','Uf','Uc','Uo','bi','bf','bc','bo']
            s_weights = [layer.name + suf for suf in s_weights]
            s_weights = ",".join(s_weights)
            s = 'layer' + str(count) + 'out = lstm(layer' + str(count-1) + 'out,' + s_weights + ');'
            file.write(s + '\n')
        count += 1
    file.write('prediction = layer' + str(len(model.layers[1:])) + 'out; \n end') 
    print('Done. Output is in file ' + name + '.m')


def keras2matlab(model_filepath,function_name):
    matlab_filepath = './' + function_name + '.m'
    file = open(matlab_filepath,"w+")
    print('Loading Model file')
    model = keras.models.load_model(model_filepath)
    model2matlab(model,file,function_name)
    file.close()



if __name__ == '__main__':
    main(sys.argv[1:])
