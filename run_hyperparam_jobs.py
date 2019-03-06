from helpers.hyperparam_helpers import make_folder_contents
import os

input_conf='configs/lstm_cnn_merge.yaml'
input_script='scripts/lstm_cnn_merge.sh'

#########################################################
## Just change the information here ##
#########################################################
subfolder='mar2'
changes_array=[["model","dense_final_activation","relu"]]
new_dirname='no_layers'
#########################################################
## No other changes required ##
#########################################################

subfolder='mar3_fixed_final_dens'

#all_changes = [(x,y,z) for x in [1,2,4] for y in ['linear','relu'] for z in ['linear','relu']]
all_changes = [(x,y,z) for x in [0,2,4] for y in ['linear','relu'] for z in [20,40]]

for num_final_layers, dens_final_act, dens_final_size in all_changes:
    new_dirname = 'numlayers_{}_densact_{}_denssize_{}'.format(num_final_layers,dens_final_act,dens_final_size)
    changes_array = [['model','num_final_layers',num_final_layers],
                     ['model','dense_final_activation',dens_final_act],
                     ['model','dense_final_size',dens_final_size]]
    output_dir=os.path.join('/global/cscratch1/sd/abbatej/autoruns',subfolder,new_dirname)
    make_folder_contents(input_conf, input_script, output_dir, changes_array)
    os.system('sbatch {}'.format(os.path.join(output_dir,'driver.sh')))

