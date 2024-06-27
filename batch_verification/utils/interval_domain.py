from .abstract_interpretation import AbstractDomain

class BoxDomain(AbstractDomain):
    def __init__(self):
        super().__init__()

    def abstract_transformer(self):
        pass

    def add(self):
        pass

    def substract(self):
        pass

    def multiply(self):
        pass

    def meet(self):
        pass

    def join(self):
        pass