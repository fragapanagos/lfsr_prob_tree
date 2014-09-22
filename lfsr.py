import numpy as np


class LFSR(object):
    taps = {
        2: (2, 1),
        3: (3, 2),
        4: (4, 3),
        5: (5, 3),
        6: (6, 5),
        7: (7, 6),
        8: (8, 6, 5, 4),
        9: (9, 5),
        10: (10, 7),
        11: (11, 9),
        12: (12, 11, 10, 4),
        13: (13, 12, 11, 8),
        14: (14, 13, 12, 2),
        15: (15, 14),
        16: (16, 14, 13, 11),
        17: (17, 14),
        18: (18, 11),
        19: (19, 18, 17, 14),
        20: (20, 17),
        21: (21, 19),
        22: (22, 21),
        23: (23, 18),
        24: (24, 23, 22, 17),
        25: (25, 22),
        26: (26, 6, 2, 1),
        27: (27, 5, 2, 1),
        28: (28, 25),
        29: (29, 27),
        30: (30, 6, 4, 1),
        31: (31, 28),
        32: (32, 22, 2, 1),
        64: (64, 63, 61, 60),
    }

    def __init__(self, nbits, seed, taps=None):
        assert nbits > 1, "must use more than 1 bit"
        self.nbits = nbits
        if taps is None:
            self.taps = LFSR.taps[nbits]
        assert seed != 0, "seed must be nonzero"
        self.state = seed

    def sample(self, nbits, rand_rolls=False):
        if rand_rolls:
            for i in xrange(np.random.randint(1, 7)):
                state = self._step()
        else:
            state = self._step()
        mask = 2**nbits-1
        return state & mask

    def _step(self):
        sr = self.state
        xor = 0
        for t in self.taps:
            xor ^= (sr >> self.nbits-t)
        xor &= 1
        self.state = (xor << self.nbits-1) | (sr >> 1)
        return self.state

    def test(self):
        """test that the lfsr taps produces a maximum length sequence"""
        seed = self.state
        ctr = 1
        while self.sample(self.nbits) != seed:
            ctr += 1
        assert ctr == 2**self.nbits-1, \
            "lfsr does not produce maximum length sequence. Produces " + \
            "length %d sequence" % ctr


def collect_lfsr_sequence(nbits, seed=0b00000001):
    """generate a max length sequence of lfsr states"""
    lfsr = LFSR(nbits, seed)
    states = []
    for i in xrange(2**nbits-1):
        states.append(lfsr.sample(nbits, rand_rolls=True))
    states = np.array(states)
    return states


def test_lfsr():
    """test that lfsr produces maximum length sequences"""
    for nbits in xrange(2, 21):
        print 'checking max length sequence for %d taps' % nbits
        lfsr = LFSR(nbits, 0b00000001)
        lfsr.test()


def test_lfsr_distribution():
    """test that the lfsr has uniform distribution"""
    for nbits in xrange(2, 20):
        states = collect_lfsr_sequence(nbits)
        unique_states = np.unique(states)
        assert len(unique_states) == len(states), \
            "%d bit lfsr does not produce uniform states." % nbits


def lfsr_autocorrelation():
    """look at the autocorrelation of the lfsr"""
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve

    fig = plt.figure(figsize=(20, 12))
    for idx, nbits in enumerate(xrange(2, 20)):
        print 'plotting autocorrelation for %d bit lfsr' % nbits
        ax = fig.add_subplot(3, 6, idx+1)
        states = collect_lfsr_sequence(nbits)
        mu = np.mean(states)
        states_0ed = states - mu
        # autocor = np.correlate(states_0ed, states_0ed, mode='full') # slow
        autocor = fftconvolve(states_0ed, states_0ed[::-1])  # way faster
        ax.plot(autocor, '-o')
        ax.set_title('%d bits' % nbits)
    fig.suptitle('autocorrelation of nbit lfsr sequences')
    plt.show()


import pytest


if __name__ == "__main__":
    test_lfsr()
    #pytest.main([__file__, '-v'])
    #lfsr_autocorrelation()
