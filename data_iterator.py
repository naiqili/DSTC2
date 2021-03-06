import numpy as np
import theano
import theano.tensor as T
import sys, getopt
import logging

# from state import *
# from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, x):
    mx = state['seqlen']
    n = state['bs']
    od = state['output_dim']
    
    X = numpy.zeros((mx, n), dtype='int32')
    Y = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32') 

    # Variable to store each utterance in reverse form (for bidirectional RNNs)
    X_reversed = numpy.zeros((mx, n), dtype='int32')
    Y_reversed = numpy.zeros((mx, n), dtype='int32')

    # Fill X and Xmask
    # Keep track of number of predictions and maximum triple length
    num_preds = 0
    num_preds_last_utterance = 0
    max_length = 0
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        triple_length = len(x[0][idx]) 

        # Fiddle-it if it is too long ..
        if mx < triple_length: 
            continue

        X[:triple_length, idx] = x[0][idx][:triple_length]
        X_reversed[:triple_length, idx] = x[0][idx][:triple_length]
        eot_indices = numpy.where(X[:, idx] == state['eot_sym'])[0]

        assert len(eot_indices) == len(x[1][idx])
        for (k, eot_idx) in enumerate(eot_indices):
            Y[eot_idx, idx] = x[1][idx][k]
            Y_reversed[eot_idx, idx] = x[1][idx][k]

        max_length = max(max_length, triple_length)

        # Set the number of predictions == sum(Xmask), for cost purposes
        num_preds += triple_length
        
        # Mark the end of phrase
        # if len(x[0][idx]) < mx:
        #     X[triple_length:, idx] = state['eot_sym'] # end of turn

        # Initialize Xmask column with ones in all positions that
        # were just set in X. 
        # Note: if we need mask to depend on tokens inside X, then we need to 
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[:triple_length, idx] = 1.

        # Reverse all utterances        
        prev_eot_index = -1
        for eot_index in eot_indices:
            X_reversed[(prev_eot_index+1):eot_index, idx] = (X_reversed[(prev_eot_index+1):eot_index, idx])[::-1]
            Y_reversed[(prev_eot_index+1):eot_index, idx] = (Y_reversed[(prev_eot_index+1):eot_index, idx])[::-1]
            prev_eot_index = eot_index
            if prev_eot_index > triple_length:
                break
     
    assert num_preds == numpy.sum(Xmask)
    
    return {'x': X,                                                 \
            'x_reversed': X_reversed,                               \
            'y': Y, \
            'y_reversed': Y_reversed,
            'mask': Xmask,                                        \
            'num_preds': num_preds,                                 \
            'num_logs': len(x[0]),                               \
            'max_length': max_length                                \
           }

class Iterator(SSIterator):
    def __init__(self, data_file, batch_size, **kwargs):
        SSIterator.__init__(self, data_file, batch_size,                   \
                            max_len=kwargs.pop('max_len', -1),               \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))
        self.k_batches = kwargs.pop('sort_k_batches', 20)
        self.state = kwargs.pop('state', None)
        # ---------------- 
        self.batch_iter = None

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch_size = self.batch_size if (batch_size == -1) else batch_size 
           
            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)
            if not len(data):
                return
            
            number_of_batches = len(data)
            # data is a set of batches
            # After chain.from_iterable, is a batch
            data = list(itertools.chain.from_iterable(data))
            
            data_x = []
            data_y = []
            for i in range(len(data)):
                data_x.append(data[i][0])
                data_y.append(data[i][1])

            x = numpy.asarray(list(itertools.chain(data_x)))
            y = numpy.asarray(list(itertools.chain(data_y)))

            lens = numpy.asarray([map(len, x)])
            order = numpy.argsort(lens.max(axis=0))
                 
            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                batch = create_padded_batch(self.state, [x[indices], y[indices]])

                if batch:
                    yield batch
    
    def start(self):
        SSIterator.start(self)
        self.batch_iter = None

    def next(self, batch_size = -1):
        """ 
        We can specify a batch size,
        independent of the object initialization. 
        """
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            return None
        return batch

def get_train_iterator(state):
    train_data = Iterator(
        state['train_file'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=True,
        max_len=state['seqlen']) 
     
    valid_data = Iterator(
        state['valid_file'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=state['seqlen'])
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_triples' in state
    test_path = state.get('test_triples')
    semantic_test_path = state.get('test_semantic', None)

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        semantic_file=semantic_test_path,
        use_infinite_loop=False,
        max_len=state['seqlen'])
    return test_data


# Test
if __name__=='__main__':
    numpy.set_printoptions(threshold='nan')
    state = {}
    state = {'bs': 7, 'seed': 1234, 'seqlen': 300, 'output_dim': 5, 'eot_sym': 1}
    state['train_file'] = './tmp/method.train'
    state['dev_file'] = './tmp/method.dev'
    train_data = Iterator(state['train_file'],
                          int(state['bs']),
                          state=state,
                          use_infinite_loop=True)
    train_data.start()

    for i in xrange(3):
        batch = train_data.next()

