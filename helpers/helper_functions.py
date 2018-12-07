import pickle
import numpy as np

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def get_rho_points(num_rho):
    core_points=np.linspace(0,.9*num_rho,25,dtype=int)
    near_edge_points=np.linspace(.9*num_rho,.95*num_rho,5,dtype=int)
    edge_points=np.linspace(.95*num_rho,num_rho,10,dtype=int)
    rho_points=np.concatenate((core_points,near_edge_points,edge_points))
    rho_points=np.unique(rho_points)
    return(rho_points)

def save_obj(obj, name):
    with open(name+'.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')
