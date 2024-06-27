from typing import Set, List
from abc import ABC, abstractmethod


class AbstractDomain(ABC):

    def __init__(self):
        self.set_concrete_elements: Set[List[List[float]]] = set() 
        self.set_abstract_elements: Set[List[List[float]]] = set()

    @abstractmethod
    def abstract_transformer():
        pass
    
    @abstractmethod
    def add():
        pass

    @abstractmethod
    def substract():
        pass

    @abstractmethod
    def multiply():
        pass

    @abstractmethod
    def meet():
        pass

    @abstractmethod
    def join():
        pass