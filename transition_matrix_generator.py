from math import log2
from math import fabs, sqrt
from itertools import chain
from numpy import zeros
import numpy as np
from random import randint, betavariate, uniform, sample
from functools import partial


class transition_matrix_generator:
    
    
    def __init__(self, n_states):
        self.n_states = n_states
    
    
    def generate_matrix(self, entropy_values):
        if len(entropy_values) != self.n_states:
            raise Exception("The length of the entropy_values should be {}".format(self.n_states))
        
        entropy2pd = partial(self.__entropy2pd, states_nr=self.n_states)
        m = list(map(entropy2pd, entropy_values))
        
        # shuffle elements within the rows
        sample_n = partial(sample, k=self.n_states)
        m_shuffled = np.array(list(map(sample_n, m)))
        
        return m_shuffled
    
    
    """
        Adapted code from: https://github.com/kkourt/entropy
        
        Use function __entropy2pd to create the row of a TPM with a certain number of states and
        whose prob distribution has a certain entropy.
    """
    
    __prob_min = 1e-6
    
    def __pd_shuffle(self, pd, times, prob_min):
        for i in range(times):
            idx0 = randint(0,len(pd)-1)
            p0 = pd.pop(idx0)
            idx1 = randint(0,len(pd)-1)
            p1 = pd.pop(idx1)
            s = p0 + p1
            # a = min(p0,p1)/s
            xmin = prob_min/s
            xmax = .5
            q0 = uniform(xmin,xmax)
            pd.append(q0*s)
            pd.append((1-q0)*s)
    
    
    def __plog2p(self, p):
        return p * log2(p) if p != 0.0 else 0.0
    
    
    def __entropy(self, pd):
        return -sum((self.__plog2p(p) for p in pd))
    
    
    def ___deviation(self, data):
        if not (isinstance(data, tuple) or isinstance(data, list)):
            data = tuple(data)
        m = self.__mean(data)
        dev = sqrt(float(sum(( (m-d)**2 for d in data )))/len(data))
        return dev
    
    
    def __mean(self, data):
        s = 0.0
        i = 0
        for item in data:
            s += item
            i += 1
        return s/i
    
    
    def __mean___entropy(self, pd_fn, pd_fn_args, times):
        return self.__mean((self.__entropy(pd_fn(*pd_fn_args)) for i in range(times) ))
    
    
    def __de(self, a, b, sum):
        return sum*(self.__plog2p(a) + self.__plog2p(1-a) - self.__plog2p(b) - self.__plog2p(1-b))
    
    
    def __pd_en_max(self, symbols_nr):
        symbols_nr_fl = float(symbols_nr)
        return (( 1.0/symbols_nr_fl for i in range(symbols_nr) ))
    
    
    def __entropy_max(self, symbols_nr):
        return self.__entropy(self.__pd_en_max(symbols_nr))
    
    
    def __pd_en_min(self, symbols_nr, prob_min):
        prob_max = 1 - ((symbols_nr-1)*prob_min)
        return chain((prob_max, ), (prob_min for i in range(symbols_nr-1)))
    
    
    def __entropy_min(self, symbols_nr, prob_min):
        return self.__entropy(self.__pd_en_min(symbols_nr, prob_min))
    
    
    def __e2pd_initial_pd(self, symbols_nr, prob_min=__prob_min, shuffle=0, initial="max"):
        if initial == "min":
            pd = list(self.__pd_en_min(symbols_nr, prob_min))
        elif initial == "max":
             pd = list(self.__pd_en_max(symbols_nr))
        else:
            raise ValueError("initial {} unknown".format(initial))
        self.__pd_shuffle(pd, shuffle, prob_min)
        pd.sort()
        return pd
    
    
    def __de_max(self, a, s):
        return self.__de(a, .5, s)
    
    
    def __de_min(self, a, s, pmin):
        return self.__de(a, pmin/s, s)
    
    
    def __de_solve(self, x, s, v, err=1e-15, limit_iterations=1000000):
        if x > .5:
            x = s - x
        if v > 0:
            ymin = x
            ymax = .5
        else:
            ymin = 0
            ymax = .5
    
        iterations = 0
        while True:
            ymed = (ymax - ymin)/2.0 + ymin
            yval = self.__de(x, ymed, s)
            if fabs(yval - v) < err:
                return ymed
            elif v > yval:
                ymin = ymed
            else:
                ymax = ymed
    
            iterations += 1
            if iterations > limit_iterations:
                raise ValueError("Unable to find a solution (min={} max={} val={} sol={})".format(ymin,ymax,yval,v))

    
    def __entropy2pd(self, tentropy, states_nr, pd=None, prob_min=__prob_min, entropy_err=.005):
        """ 
            Create a probability distribution (pd) for a set of states that will
            adhere to the given entropy value
    
            tentropy    : target entropy value
            symbols_nr  : number of symbols
            prob_min    : minum value for probabilities
            entropy_err : margin for error between the given and pd entropy
    
            The basic concept is to choose two probabilities from the list and
            modify them so that we are close to the target entropy.
    
            returns a list of symbols_nr probabilities
        """
        if tentropy == 0:
            l = zeros(states_nr)
            l[states_nr - 1] = 1
            return list(l)
        
        # sanity checks
        if (tentropy > self.__entropy_max(states_nr)):
            raise ValueError("entropy specified ({}) is too high".format(tentropy))
        if (tentropy < self.__entropy_min(states_nr, prob_min)):
            raise ValueError("entropy specified ({}) is too small".format(tentropy))
    
        # Choose an initial probability distribution
        if pd is None:
            pd = self.__e2pd_initial_pd(states_nr, prob_min)
        iterations = 0
        alpha = beta = 1.3
        while True:
            pd.sort()
            #if iterations % 7 == 0: bm.add("%013d" % iterations, pd )
            entropy_pd = self.__entropy(pd)
            de = tentropy - entropy_pd
            if iterations % 512 == 1: 
                alpha = uniform(.5, 10)
                beta  = uniform(.5, 10)
            if fabs(de) <= entropy_err:
                break
            elif tentropy > entropy_pd:
                #print 'INC '
                p0 = pd.pop(int((len(pd))*(betavariate(alpha,beta))))
                p1 = pd.pop(int((len(pd))*(betavariate(beta,alpha))))
                s = p0 + p1
                a = min(p0,p1)/s
                #assert a<.5
                xmin = a
                xmax = .5
            else:
                #print 'DEC '
                p0 = pd.pop(int((len(pd))*(betavariate(alpha,beta))))
                p1 = pd.pop(int((len(pd))*(betavariate(alpha,beta))))
                s = p0 + p1
                a = min(p0,p1)/s
                xmin = prob_min/s
                xmax = a
    
            q0 = uniform(xmin,xmax)
            pd.append(q0*s)
            pd.append((1-q0)*s)
    
            iterations += 1
    
        return pd
