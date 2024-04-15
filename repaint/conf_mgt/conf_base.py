import torch

from collections import defaultdict

class NoneDict(defaultdict):
    def __init__(self):
        super().__init__(self.return_None)

    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


class Default_Conf(NoneDict):
    def __init__(self):
        pass

    @staticmethod
    def device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
