import sys
from typing import Set, List, Any
from abc import ABC, abstractmethod


class AbstractDomain(ABC):

    def __init__(self, _lower_bound: float = sys.float_info.min, _upper_bound: float = sys.float_info.max):
        if _lower_bound > _upper_bound:
            raise ValueError("lower bound should be less than or equal to upper bound.")

        self.id: int = -1
        self.name: str = "default"
        self.lower_bound: float = _lower_bound
        self.upper_bound: float = _upper_bound
        
        
    
    @classmethod
    def from_lower_bound(cls, _lower_bound: float):
        return cls(_lower_bound=_lower_bound)

    @classmethod
    def from_upper_bound(cls, _upper_bound: float):
        return cls(_upper_bound=_upper_bound)
    
    @classmethod
    def from_lower_and_upper_bound(cls, _lower_bound: float, _upper_bound: float):
        return cls(_lower_bound=_lower_bound, _upper_bound=_upper_bound)
    
    @abstractmethod
    def abstract_transformer(self):
        pass
    
    @abstractmethod
    def add(self, other: Any) -> Any:
        pass

    @abstractmethod
    def substract(self, other: Any) -> Any:
        pass

    @abstractmethod
    def multiply(self, other: Any) -> Any:
        pass

    @abstractmethod
    def meet(self, other: Any) -> Any:
        pass

    @abstractmethod
    def join(self, other: Any) -> Any:
        pass