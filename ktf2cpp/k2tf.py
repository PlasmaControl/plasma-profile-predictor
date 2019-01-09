"""
k2tf.py
Converts a keras hdf5 model to tensorflow pb model.
Author: Rory Conlin
Date: 1/9/19
"""

import keras
import tensorflow as tf
import argparse
from pathlib import Path
import sys


def parse_args(args):
    """Parses command line arguments

    Args:
        args (list): List of arguments to parse, ie sys.argv

    Returns:
        parsed_args (argparse object): argparse object containing parsed
        arguments.
    """

    parser = argparse.ArgumentParser(prog='convert_k2tf',
                                     description="""converts a keras hdf5 model (.h5)  file to tensorflow protocol buffer (.pb)""")
    parser.add_argument("input_path",
                        help="file path to keras model file")
    parser.add_argument("-o", "--output_path",
                        help="""Path and filename to save output. Default is current directory, same name, but .pb instead of .h5 """, metavar='')
    parser.add_argument(
        "-p", "--prefix", help="""prefix to use for output node names""", metavar='')
    return parser.parse_args(args)


def convert_k2tf(model_path, out_path, output_node_prefix=None):
    """Converts a keras model to tensorflow


    """
    print(model_path)
    print(out_path)
    if out_path:
        out_path_abs = Path(out_path).resolve()
        out_path_fld = str(out_path_abs.parent)
        out_path_name = str(out_path_abs.name)
        out_path_stem = str(out_path_abs.stem)
    else:
        out_path_abs = Path(model_path).resolve()
        out_path_fld = str(out_path_abs.parent)
        out_path_name = str(out_path_abs.stem + '.pb')
        out_path_stem = str(out_path_abs.stem)

    model = keras.models.load_model(model_path)
    keras.backend.set_learning_phase(0)  # tell keras we're not training
    sess = keras.backend.get_session()

    output_node_names = [node.op.name for node in model.outputs]
    if output_node_prefix:
        num_output = len(output_node_names)
        pred = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            output_node_names[i] = '{}{}'.format(
                output_node_prefix, i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=output_node_names[i])

    print('Output nodes names are: ', output_node_names)

    # Write the graph in binary .pb file
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_node_names)
    tf.io.write_graph(constant_graph, out_path_fld,
                      out_path_name, as_text=False)
    print('Saved the constant graph (ready for inference) at: ',
          str(Path(out_path_fld) / out_path_name))


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    convert_k2tf(args.input_path, args.output_path, args.prefix)
