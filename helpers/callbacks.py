from time import time
import keras
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np


class DynamicWeighting(Callback):
    """Callback that updates weights of different losses to maintain 
    balance for multi-output models.
    """

    def __init__(self,weights,metrics,mode='epoch'):
        for weight in weights:
            if type(weight) is not type(K.variable(1)):
                raise TypeError("All weights for dynamic weighting must be keras tensors. Got {}".format(type(weight)))
        assert len(weights) == len(metrics), "Each weight must have a corresponding metric"
        self.weights = weights
        self.metrics = metrics
        self.mode = mode
        self.total_weight = np.sum([K.eval(weight) for weight in self.weights])
        
    def on_batch_end(self,batch,logs={}):
        if self.mode == 'batch':
            losses = [logs.get(metric,0) for metric in self.metrics]
            total_loss = np.sum(losses)
            if total_loss >0:
                for weight, loss in zip(weights,losses):
                    K.set_value(weight, loss/total_loss)
        else:
            pass
        
    def on_epoch_end(self,epoch,logs):
        if self.mode == 'epoch':
            losses = [logs.get(metric,0) for metric in self.metrics]
            total_loss = np.sum(losses)
            if total_loss >0:
                for weight, loss in zip(weights,losses):
                    K.set_value(weight, self.total_weight*loss/total_loss)
        else:
            pass    
            
            
class TimingCallback(Callback):
    """A Keras Callback which records the time of each epoch
    
    Can also set a maximum training time, and training will stop 
    early if anticipated time for the next epoch exceeds time remaining.
    Useful when training on clusters where computation time must be limited in advance,
    so your job can exit cleanly rather than being killed.
    """

    def __init__(self, time_limit=np.inf):
        self.epoch_times = []
        self.start_times = []
        self.end_times = []
        self.mean_time = 0
        self.std_time = 0
        self.cum_time = 0
        self.time_limit = time_limit

    def on_epoch_begin(self, epoch, logs={}):
        self.start_times.append(time())
        if 'start_times' in logs:
            logs['start_times'].append(self.start_times[-1])
        else:
            logs['start_times'] = [self.start_times[-1]]

    def on_epoch_end(self, epoch, logs={}):
        self.end_times.append(time())
        if 'end_times' in logs:
            logs['end_times'].append(self.end_times[-1])
        else:
            logs['end_times'] = [self.end_times[-1]]
            
        self.epoch_times.append(self.end_times[-1] - self.start_times[-1])
        if 'epoch_times' in logs:
            logs['epoch_times'].append(self.epoch_times[-1])
        else:
            logs['epoch_times'] = [self.epoch_times[-1]]
        
        self.mean_time = np.mean(self.epoch_times)
        self.std_time = np.std(self.epoch_times)
        self.cum_time = np.sum(self.epoch_times)
        
        if self.cum_time + self.mean_time + 3*self.std_time > self.time_limit:
            print('Stopping early due to time constraint')
            self.stopped_epoch = epoch
            self.model.stop_training = True
            
class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback
    when training with a generator.
    Basically just grabs all the data from the generator and puts it in one array.
    """

    def __init__(self, generator, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.generator = generator
        self.gen_steps = len(self.generator)
        self.profile_inputs = self.generator.profile_inputs
        self.actuator_inputs = self.generator.actuator_inputs
        self.target_names = self.generator.targets
        self.batch_size = self.generator.batch_size
        self.sample_weights = np.ones(self.gen_steps*self.batch_size)

    def on_epoch_end(self, epoch, logs):
        inputs = {}
        targets = {}
        for sig in self.profile_inputs:
            inputs['input_' + sig] = []
        for sig in self.actuator_inputs:
            inputs['input_future_' + sig] = []
            inputs['input_past_' + sig] = []
        for sig in self.target_names:
            targets['target_' + sig] = []
        for s in range(self.gen_steps):
            inp, targ = self.generator[s]
            for sig in self.profile_inputs:
                inputs['input_' + sig].append(inp['input_' + sig])
            for sig in self.actuator_inputs:
                inputs['input_past_' + sig].append(inp['input_past_' + sig])
                inputs['input_future_' +
                       sig].append(inp['input_future_' + sig])
            for sig in self.target_names:
                targets['target_' + sig].append(targ['target_' + sig])

        for key in inputs.keys():
            inputs[key] = np.concatenate(inputs[key], axis=0)
        for key in targets.keys():
            targets[key] = np.concatenate(targets[key], axis=0)

        self.validation_data = [inputs['input_' + sig] for sig in self.profile_inputs] + \
                               [inputs['input_past_' + sig] for sig in self.actuator_inputs] + \
                               [inputs['input_future_' + sig] for sig in self.actuator_inputs] + \
                               [targets['target_' + sig]
                                   for sig in self.target_names] + [self.sample_weights for _ in range(len(self.target_names))]

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
