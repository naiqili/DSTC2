import theano
import theano.tensor as T
import numpy as np
from utils import *

m = np.asarray([[0,0,1],
                [0,1,0]])

x = T.matrix('x')
y = SoftMax(x)
f = theano.function([x], y)

print f(m)
