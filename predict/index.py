from diamond import Diamond, Diamond4P
import math
from collections import OrderedDict

class DiamondIndex:
    def __init__(self, max_size = 100, k=12):
        self.index = OrderedDict()
        self.max_size = max_size
        self.k = k
        
    def add(self, diamond: Diamond) -> None:
        if len(self.index) >= self.max_size:
            self.index.popitem(last=False)
        self.index[diamond] = math.ceil(diamond.predict_price(k=self.k))
        
    def get(self, diamond: Diamond) -> float:
        if diamond not in self.index:
            self.add(diamond)
        return self.index[diamond]
    
    
class DiamondIndex4P:
    def __init__(self, max_size = 100, k=10):
        self.index = OrderedDict()
        self.max_size = max_size
        self.k = k
        
    def add(self, diamond: Diamond4P) -> None:
        if len(self.index) >= self.max_size:
            self.index.popitem(last=False)
        self.index[diamond] = math.ceil(diamond.predict_price(k=self.k))
        
    def get(self, diamond: Diamond4P) -> float:
        if diamond not in self.index:
            self.add(diamond)
        return self.index[diamond]