import numpy as np
import random

class DidacticSamplesGenerator():
    """
    Generate samples defined by some structure and values
    """

    def __init__(self, structure, values):
        self.structure = structure
        self.values = values

    def opsnum_compatible(self, n_range):
        n_terms = len(self.structure[:-1].split('+'))
        return n_range[0] <= n_terms < n_range[1]

    def __call__(self, s_range):
        terms = self.structure[:-1].split('+')
        sections = [t.split('*') for t in terms]
        sequence = []
        curr_numsize = np.random.randint(*s_range)
        for t in sections:
            usable_size = curr_numsize-len(t[-1])
            sec_num = len(t) - 1
            seq = ''
            for i, s in enumerate(t):
                pattern = s[:-1] if i < len(t)-1 else s
                new_seq = ''
                for c in pattern:
                    new_seq += ''.join(
                        np.random.choice(self.values[c], size=1))
                if i < len(t)-1:
                    remaining_size = max(
                        0, (usable_size // sec_num) - len(new_seq))
                    new_seq += ''.join(np.random.choice(
                        self.values[s[-1]], size=remaining_size))
                usable_size -= len(new_seq)
                sec_num -= 1
                seq += new_seq

            sequence.append(str(int(seq)))

        return sequence


class GeneratorsCollection():
  def __init__(self, generators):
    self.generators = generators

  def __call__(self, n_range, s_range):
    compatible_gens = []
    for gen in self.generators:
      if gen.opsnum_compatible(n_range):
        compatible_gens.append(gen)
    
    return random.choice(compatible_gens)(s_range) if len(compatible_gens) > 0 else None
