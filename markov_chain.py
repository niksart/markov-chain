import numpy as np

""" 
  constructor:
    - tpm: transition probability matrix (numpy)
    - init_distr: initial distribution of the probabilities of the states
    - state_labels: the text associated to each state
"""
class markov_chain:
  
  
  def __init__(self, tpm, init_distr, state_labels=None):
    if tpm.shape[0] != tpm.shape[1]:
      raise Exception("The transition probability matrix should be a squared matrix.")
    
    # get the number of states
    self.n_states = tpm.shape[0]
    
    # transition probability matrix
    self.tpm = tpm
    
    # initial distribution
    if len(init_distr) != self.n_states:
      raise Exception("Each state should have an initial probability.")
    self.init_distr = init_distr
    
    # save the state labels
    if state_labels != None:
      if len(state_labels) == self.n_states:
        # each label should be string
        for l in state_labels:
          if type(l) != str:
            raise Exception("Each label should be a string.")
        
        self.state_labels = state_labels
      else:
        raise Exception("If labeled, each state should have a label.")
    else:
      self.state_labels = None
  
  
  def generate_text(self, steps):
    if self.state_labels != None:
      # choose the starting state
      current_state = np.random.choice(self.n_states, 1, p=self.init_distr)[0]
      
      text = ""
      for _ in range(steps):
        text += self.state_labels[current_state]
        current_state = self.choose_state(current_state)
      
      return text
    
    else:
      raise Exception("State label should be set with the constructor for generating text.")
  
  
  def choose_state(self, state):
    """ Function that returns the number of the chosen state """
    distr = self.tpm[state]
    chosen_state = np.random.choice(self.n_states, 1, p=distr)[0]
    return chosen_state
  
  
  def tpm_after_n_steps(self, n_steps):
    """ Function that performs the n times the product of the tpm matrix """
    tpm = self.tpm
    for _ in range(n_steps - 1):
      tpm = np.dot(tpm, self.tpm)
    return tpm
  
  