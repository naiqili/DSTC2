# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from encdec import *
from utils import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import pprint
import numpy
import collections
import signal
import math


import matplotlib
matplotlib.use('Agg')
import pylab


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('./log/' + __name__))

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "valid_cost"]

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings):
    print("Saving the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print("Model saved, took {}".format(time.time() - start))

def load(model, filename):
    print("Loading the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print("Model loaded, took {}".format(time.time() - start))

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    state = eval(args.prototype)() 
    timings = init_timings() 
    
    
    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    model = EncoderDecoder(state)
    rng = model.rng 

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    logger.debug("Compile trainer")
    logger.debug("Training with exact log-likelihood")
    train_batch = model.build_train_function()

    eval_batch = model.build_eval_function()
    # eval_misclass_batch = model.build_eval_misclassification_function()

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()
    
    # Start looping through the dataset
    step = -1
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_misclass = 0
    train_done = 0
    ex_done = 0
     
    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        step = step + 1
        # Sample stuff
        if step % 200 == 0:
            for param in model.params:
                logger.debug("%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5))

        # Training phase
        batch = train_data.next() 

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        logger.debug("[TRAIN] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
        x_data = batch['x']
        y_data = batch['y']
        # max_length = batch['max_length']
        # x_cost_mask = batch['x_mask']
        # x_semantic = batch['x_semantic']

        c = train_batch(x_data, y_data)

        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost = c

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)
            logger.debug(".. %.2d:%.2d:%.2d %4d mb # %d bs %d cost = %.4f" % (h, m, s,\
                                                                 state['time_stop'] - (time.time() - start_time)/60.,\
                                                                 step, \
                                                                 batch['x'].shape[1], \
                                                                 float(c)))
        
        if valid_data is not None and step % state['valid_freq'] == 0 and step > 1:
            valid_data.start()

            logger.debug("[VALIDATION START]")
            vcost_list = []
                
            while True:
                batch = valid_data.next()
                # Train finished
                if not batch:
                    break
                logger.debug("[VALID] - Got batch %d" % (batch['x'].shape[1]))
                        
                x_data = batch['x']
                y_data = batch['y']

                c = eval_batch(x_data, y_data)

                if numpy.isinf(c) or numpy.isnan(c):
                    continue
                        
                vcost_list.append(c)
                
            logger.debug("[VALIDATION END]") 
                
            valid_cost = numpy.mean(vcost_list)

            if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                patience = state['patience']
                # Saving model if decrease in validation cost
                save(model, timings)
            elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                patience -= 1

            timings["train_cost"].append(train_cost)
            timings["valid_cost"].append(valid_cost)

            # Reset train cost, train misclass and train done
            train_cost = 0

            logger.debug("[VALIDATION COST]: %f" % valid_cost)

            # Plot histogram over validation costs
            try:
                pylab.figure()
                pylab.subplot(2,1,1)
                pylab.title("Training Cost")
                pylab.plot(timings["train_cost"])
                pylab.subplot(2,1,2)
                pylab.title("Validation Cost")
                pylab.plot(timings["valid_cost"])
                pylab.savefig(model.state['save_dir'] + '/' + str(step) + '.png')
            except:
                pass

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
