import theano
import theano.tensor as T
import numpy as np

x = T.tensor3('x')
y = T.imatrix('y')

m_x = np.arange(27).reshape((3,3,3))
print "mx:", m_x
m_y = [[0,1,2],
       [1,2,0],
       [2,0,1]]
print "my:", m_y

x_flatten = x.dimshuffle(2,0,1).flatten(2).dimshuffle(1,0)
y_flatten = y.flatten()

res = x_flatten[T.arange(y_flatten.shape[0]), y_flatten]

f = theano.function([x, y], res)

print "res:", f(m_x, m_y)

reshape_res = res.reshape(y.shape)
f2 = theano.function([x, y], reshape_res)

print "reshape:", f2(m_x, m_y)
