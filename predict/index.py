from diamond import Diamond, Diamond4P

class DiamondIndex:
    def __init__(self, max_size = 100, k=12):
        self.index: dict[Diamond, dict] = {}
        self.max_size = max_size
        self.k = k
        
    def add(self, diamond: Diamond) -> None:
        if len(self.index) >= self.max_size:
            self.index.popitem(last=False)
        self.index[diamond] = diamond.predict_price(k=self.k)
        
    def get(self, diamond: Diamond) -> dict:
        if diamond not in self.index:
            self.add(diamond)
        return self.index[diamond]
    
    
class DiamondIndex4P:
    def __init__(self, max_size = 100, k=10):
        self.index: dict[Diamond4P, dict] = {}
        self.max_size = max_size
        self.k = k
        
    def add(self, diamond: Diamond4P) -> None:
        if len(self.index) >= self.max_size:
            self.index.popitem(last=False)
        self.index[diamond] = diamond.predict_price(k=self.k)
        
    def get(self, diamond: Diamond4P) -> dict:
        if diamond not in self.index:
            self.add(diamond)
        return self.index[diamond]