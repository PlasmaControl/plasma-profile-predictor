from math import ceil

def exp(initial_learning_rate, decay_steps, decay_rate, staircase = False):
    '''
    Exponential decay at in learning rate.

    Args:
    decay_steps: Number of steps until specified decay
    decay_rate: rate of decay
    stiarcase: when set to True, decay only applied after a period of decay_steps
    '''
    def scheduler(epoch):
        if staircase:
            return initial_learning_rate * decay_rate ** (int(epoch/decay_steps))
        else:
            return initial_learning_rate * decay_rate ** (epoch/decay_steps)
    return scheduler

def piece(boundaries, values):
    '''
    Piecewise decay. 
    
    Args:
    boundaries(list): how many steps at specified decay rate
    values(list): decay rates. Length must be length of boundaries + 1
    '''
    def scheduler(epoch):
        # Binary search for interval
        low = 0
        high = len(boundaries)-1
        if epoch > boundaries[high]:
            return values[high+1]
        mid = int((low+high)/2)
        while low < high:
            mid = int((low+high)/2)
            if epoch < boundaries[mid]:
                high = mid
            elif epoch > boundaries[mid] and low != mid:
                low = mid
            elif epoch == boundaries[mid]:
                high = mid
                break
            else:
                low = mid+1
        return values[high]
    return scheduler

def poly(initial_learning_rate, decay_steps, end_learning_rate, power=1.0, cycle=False):
    '''
    Polynomial decay
    
    Args: self-explanatory
    '''
    def scheduler(epoch):
        if not cycle:
            epoch = min(epoch, decay_steps)
            return ((initial_learning_rate - end_learning_rate) *
                    (1 - epoch / decay_steps) ** (power)) + end_learning_rate
        else:
            decay_steps_1 = decay_steps * ceil(epoch / decay_steps)
            return ((initial_learning_rate - end_learning_rate) *
                    (1 - epoch / decay_steps_1) ** (power)) + end_learning_rate
        return scheduler

def decayed_learning_rate(initial_learning_rate, decay_steps, decay_rate, staircase=False):
    '''
    Inverse time decay
    '''
    def scheduler(epoch):
        if staircase:
            return initial_learning_rate / (1 + decay_rate * math.floor(epoch / decay_step))
        else:
            return initial_learning_rate / (1 + decay_rate * epoch / decay_step)
        return scheduler


    
