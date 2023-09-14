"""Module containing methods to scramble labels in tensors"""
import numpy as np
import torch


class RandomToyboxScrambler:
    """Class definition for label scrambler that randomly scrambles labels for Toybox dataset"""
    
    def __init__(self, seed = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.labels = np.arange(12)
        self.rng.shuffle(self.labels)
        self.shift = {}
        for i, l in enumerate(self.labels):
            self.shift[i] = l
        self.used = False
    
    def get_shift(self):
        """Return shift dictionary"""
        return self.shift
    
    def print_shift(self):
        """Print the shift dictionary"""
        for i, l in self.shift.items():
            print("{} -> {}".format(i, l))
        
    def scramble(self, labels: torch.tensor):
        """Scramble the labels provided"""
        scrambled_labels = labels.clone().detach().cpu()
        scrambled_labels.apply_(lambda x: self.labels[x])
        if not self.used:
            print("Using {}...".format(self.__class__.__name__))
            self.used = True
        return scrambled_labels.cuda()
        
        
def test():
    scrambler = RandomToyboxScrambler(seed=5)
    labels = torch.randint(low=0, high=12, size=[10])
    scrambler.print_shift()
    print(labels)
    print(scrambler.scramble(labels))
    
    
if __name__ == "__main__":
    test()
    