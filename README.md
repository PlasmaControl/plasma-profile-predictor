# Plasma Profile Predictor


## Repository layout


### `root/`

`combine_data.py`: merge multiple smaller datasets into one larger one. Mostly used by Joe.

`ensemble_evaluate.py`: Load a bunch of models and evaluate them as an ensemble model

`evaluate.py`: Run after training to compute a bunch of metrics that can't be computed during training. Uses the same metrics/data for each model so an apples to apples comparison can be made.

`train_traverse.py`: Main training script. 

* Run without arguments, (`python train_traverse.py`) it runs the default "scenario" laid out at the beginning of the script. 

* Running with an argument of `-1` (`python train_traverse.py -1`) submits batch jobs to run a hyperparameter scan.

* After training, outputs a model file (`.h5`) and parameter dictionary (`.pkl`) for each run. The parameter dictionary contains all the settings and training results from the run for analysis later.

### `helpers/`

`callbacks.py`: Some custom callbacks used during training.

`custom_losses.py`: custom loss functions, though we've tended to stick with MSE or MAE lately.

`data_generator.py`: function for processing data to get ready for training, and class for a data generator to provide data during training.

`exclude_shots.py`: list of shot numbers broken up by shot topology, in general we exclude shots if they have a "non standard" topology (ie, not enough training samples).

`hyperparam_helpers.py`: functions to generate and submit slurm scripts for doing hyperparameter scans.

`normalization.py`: functions for normalizing, denormalizing, and renormalizing data. Methods:
*	`StandardScaler`: scales to mean 0, std 1 by subtracting mean and dividing by standard deviation
*	`MinMax`: scale to between 0 and 1 by subtracting min and dividing by (max-min)
*	`MaxAbs`: scales to between -1 and 1 by dividing by max absolute value
*	`RobustScaler`: scales to approximately mean 0 std 1, by subtracting median and dividing by inter-quartile range
*	`PowerTransform`: nonlinear scaling via Yeo-Johnson transform

`profile_fitting.py`: some old stuff for fitting profiles from raw data for real time use.

`pruning_functions.py`: functions to remove samples from training set based on various criteria, eg:
*	`remove_dudtrip`: remove any samples during or after a disruption or PCS crash etc.
*	`remove_non_gas_feedback`: remove samples where gas feedback was not used to control density
*	`remove_non_beta_feedback`: remove samples where beta feedback was not used to control beams
*	`remove_I_coil`: remove samples during and after non-standard 3d coil operation (ie, field perturbations etc)
*	`remove_gas`: remove samples where gas other than H/D/T were used.
*	`remove_ECH`: remove samples where ECH was used
*	`remove_nan`: remove samples where any value is nan. (otherwise nan values will get replaced with a mean value during normalization)

`results_processing.py`: some functions to automatically analyze models and generate some plots, though we mostly use notebooks for this now

### `models/`

Basically a bunch of functions for creating trainable models. The main ones we use are in `LSTMConv1D.py` and `LSTMConv2D.py`

The basic API is that each function should take the following arguments:
*	`input_profile_names`: list of names of the profiles used as inputs (eg `temp`, `dens`, `press_EFIT01`)
*	`target_profile_names`: list of names of profiles the model should predict  (eg `temp`, `dens`, `press_EFIT01`)
*	`scalar_input_names`: list of names of scalar signals (eg `kappa_EFIT01`, `triangularity_top_EFITRT1`)
*	`actuator_names`: list of names of actuators used by the model (eg `pinj`, `curr_target`)
*	`lookbacks`: dictionary of lookback values. Keys should be the names of each signal, values should be integers of how many steps to use for lookback.
*	`lookahead`: how many steps in the future to predict.
*	`profile_length`: integer, size of the spatial grid the profiles are discretized on (ie, size of the input arrays).
*	`std_activation`: name of the default activation function to use in the model (eg, `relu`, `elu`, `tanh`)
*	`**kwargs`: keyword arugments passed to the model constructor. Common ones include:
	*	`max_channels`: number of channels to use at the widest point in the model.
	*	`l2`: coefficient for L2 regularization on model weights.
	*	`kernel_init`: name of initializer to use for kernel weights.
	*	`bias_init`: name of initializer to use for bias weights.

Each constructor function should return a keras model. To work with the rest of the pipeline, the model should obey the following rules:
*	Each input signal has its own input layer, and the layer should be named `input_<signal name>`. 
*	Each output should have its own layer, named `target_<signal name>`
*	If the model is intended to be used on the real time control system, try to stick to layer types / options supported by keras2c: https://github.com/f0uriest/keras2c
	
### `notebooks/`
A bunch of jupyter notebooks used for plotting, analyzing results, examining data, prototyping new functions etc. Too many to list in detail, a lot haven't been used in a while, just look at the timestamps to see whats been used recently and use that as a template.

## General Linux helpers:

* `cd path/to/file` change directory
* `mkdir newfolder`: create a new folder/directory
* `ls`: print a list of files in the current directory
* `ll`: print list of files with details
* `du -d 1`: show size of files/folders
* `mv path/from path/to`: move or rename files
* `cp path/from path/to`: copy files
* `scp path/from path/to`: secure copy over SSH for moving stuff to and from traverse
* `ssh-copy-id netID@traverse.princeton.edu`: copy ssh key so you dont have to type your password any time you connect
* `slurmtop`: show currently running jobs and node usage. Use with the `-u netID` flag to only see your jobs.


Linux command line intro:
http://linuxcommand.org/lc3_learning_the_shell.php

Git intro (first 3 chapters):
https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control



## Getting started on Traverse:

Connect to the princeton VPN

connect:

`ssh netID@traverse.princeton.edu`

clone this repository 

load anaconda:

`module load anaconda`

create new environment using the config file in the headnode of this repository:

`conda create --name tfgpu --file conda-pkg-list.txt`


### Connecting to Traverse and using Jupyter lab:

From your local machine:

`ssh -N -f -L localhost:8893:localhost:8893 netID@traverse.princeton.edu`

this will open a connection to port 8893 on traverse. You sometimes might get an error saying that port is in use, in which case just change 8893 to another one, usually 8892, 8893, 8894, 8895 etc

Then connect to traverse:

`ssh netID@traverse.princeton.edu`

Load anaconda and activate the environment:

`module load anaconda`

`conda activate tfgpu`

Start jupyter:

`jupyter lab --no-browser --port=8893 --ip=127.0.0.1`

this will start a jupyter server using port 8893 (if you used a different port above change it here too)
You should then see something like:
```
To access the notebook, open this file in a browser:
        file:///home/wconlin/.local/share/jupyter/runtime/nbserver-83045-open.html
    Or copy and paste one of these URLs:
        http://127.0.0.1:8894/?token=6af4e8b5530a3d0b28d3d1c1038b306486840accd5b7802f
     or http://127.0.0.1:8894/?token=6af4e8b5530a3d0b28d3d1c1038b306486840accd5b7802f
```

copy one of the last 2 links (starts with http://127....) and paste it in your local browser, and that should connect to jupyter lab.

Further jupyter lab info:

https://jupyterlab.readthedocs.io/en/stable/

Project directory on traverse, where we store raw data and results of training:

`/projects/EKOLEMEN/profile_predictor`
