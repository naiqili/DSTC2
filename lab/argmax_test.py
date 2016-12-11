import theano
import theano.tensor as T
import numpy as np

m = np.random.rand(3,3,3)
x = T.tensor3('x')
y = x.argmax(axis=2)
print "given matrix:"
print m
print

f = theano.function([x], y)
print "argmax along axis=2"
print f(m)
