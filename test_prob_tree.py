import numpy as np
from prob_tree import ProbTree
import matplotlib.pyplot as plt


def ptree_experiment(w_nbits, lfsr_nbits, dist, nsamples=100000, title=None):
    assert lfsr_nbits >= w_nbits

    ptree = ProbTree(w_nbits, lfsr_nbits, lfsr_seed=0b00101010, w_dist=dist)
    samples = np.zeros(nsamples, dtype=int)
    for i in xrange(nsamples):
        samples[i] = ptree.sample()
    bincount = np.bincount(samples)
    w_emp = np.zeros(ptree.w.shape)
    w_emp[:len(bincount)] = bincount/float(nsamples)

    apx_error = abs(ptree.w - ptree.w_apx)
    emp_error = abs(ptree.w - w_emp)
    q_level = 1./2**(w_nbits+1)

    fig = plt.figure(figsize=(8, 10))

    ax = fig.add_subplot(211)
    ax.plot(ptree.w, 'bo', label='desired')
    #ax.plot(ptree.w_apx, 'ro', label='tree approximation')
    ax.plot(w_emp, 'go', label='measured')
    ax.legend(loc='best')
    ax.set_title('probability distributions')
    ax.set_xlim(0, len(ptree.w))
    ax.set_xticklabels([])

    ax = fig.add_subplot(212)
    #ax.plot(apx_error, 'ro')
    ax.plot(emp_error, 'go')
    ax.axhline(q_level, color='k')
    ax.set_title('distribution errors')
    ax.set_xlim(0, len(ptree.w))

    if title is not None:
        fig.suptitle(title, fontsize=18)

w_nbits = 8
lfsr_nbits = 32

# uniform distribution
#for n in [8, 32, 128, 256, 512]:
#for n in [8, 512]:
for n in [4096]:
    uniform = np.ones(n)/n
    ptree_experiment(w_nbits, lfsr_nbits, uniform, title='uniform%d' % n)

# sample from uniform distribution
N=4096
# d = np.random.rand(N)
# ptree_experiment(w_nbits, lfsr_nbits, d, title='sampled uniform')


# distributions that looks like a sin
P = 1
x = np.linspace(-P*np.pi, P*np.pi, N)
d = .5*(np.sin(x)+1)
ptree_experiment(w_nbits, lfsr_nbits, d, title='1P sine')

# P = 2
# x = np.linspace(-P*np.pi, P*np.pi, N)
# d = .5*(np.sin(x)+1)
# ptree_experiment(w_nbits, lfsr_nbits, d, title='2P sine')
 
dims = 16
D = np.random.multivariate_normal(np.ones((dims,)), np.identity(dims), N)
for i in range(N):
    D[i,:] /= np.linalg.norm(D[i,:])
d = np.abs(D[:,0])
ptree_experiment(w_nbits, lfsr_nbits, d, title='hypersphere')

plt.show()
