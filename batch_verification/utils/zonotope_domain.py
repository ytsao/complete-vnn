import sys
from typing import Any, List

from jax import numpy as jnp
from jax import vmap

from abstract_interpretation import AbstractDomain
from interval_domain import BoxDomain

class ZonotopeDomain(AbstractDomain):

    num_zonotopes: int = 0

    def __init__(self, 
                 _lower_bound: float = sys.float_info.min, 
                 _upper_bound: float = sys.float_info.max, 
                 _concrete_domain: BoxDomain = BoxDomain(),
                 _center: float = 0.0,
                 _var_generators: List[str] = [],
                 _coef_generators: List[float] = []):
        super().__init__(_lower_bound=_lower_bound, _upper_bound=_upper_bound)
        
        self.concrete_domain: BoxDomain = BoxDomain(_lower_bound=_lower_bound, _upper_bound=_upper_bound)
        self.center: float = _center
        self.var_generators: List[Any] = []
        self.coef_generators: List[float] = []
        self.name = f"zonotope_{ZonotopeDomain.num_zonotopes}"
        self.id = ZonotopeDomain.num_zonotopes

        ZonotopeDomain.num_zonotopes += 1

    @classmethod
    def with_center_and_generators(cls, _center: float, _var_generators: List[str], _coef_generators: List[float]):
        if len(_var_generators) != len(_coef_generators):
            raise ValueError("the number of variables and coefficients should be the same.")

        lb: float = _center - sum(abs(_coef_generators))
        ub: float = _center + sum(abs(_coef_generators))

        return cls(_lower_bound=lb, 
                   _upper_bound=ub,  
                   _center=_center, 
                   _var_generators=_var_generators, 
                   _coef_generators=_coef_generators)


    @classmethod
    def from_lower_and_upper_bound(cls, _lower_bound: float, _upper_bound: float):
        return cls(_lower_bound=_lower_bound, 
                   _upper_bound=_upper_bound, 
                   _center=(_lower_bound + _upper_bound) / 2.0)


    def abstract_transformer(self) -> None:
        """
        * introduce generator variables for zonotope domain.
        
        * according to "center", "lb", and "ub", we can calculate the coefficients of the generator variables.

        * only supporting from interval to zonotope in current implementation.
        ? we can extend this function to support other abstract transformer:
        ? for example: 
        ?   there are more than one epsilon variables in the zonotope domain when initialzation stage.
        """
        diviation: float = self.upper_bound - self.center
        for i in range(ZonotopeDomain.num_zonotopes):
            self.var_generators.append(f"epsilon_{i}")
            if i == self.id:
                self.coef_generators.append(diviation)
            else:
                self.coef_generators.append(0.0)

        return 


    def add(self, other: Any) -> Any:
        if len(self.var_generators) == 0 or len(other.var_generators) == 0:
            raise ValueError("There is no generator variables in zonotope domain.")

        self.center += other.center
        self.coef_generators = (jnp.array(self.coef_generators) + jnp.array(other.coef_generators)).tolist()
        # * self.var_generators remains the same
        self.concrete_domain.lower_bound = self.center - jnp.sum(jnp.abs(jnp.array(self.coef_generators)))
        self.concrete_domain.upper_bound = self.center + jnp.sum(jnp.abs(jnp.array(self.coef_generators)))
        self.lower_bound = self.concrete_domain.lower_bound
        self.upper_bound = self.concrete_domain.upper_bound

        return 


    def substract(self, other: Any) -> Any:
        if len(self.var_generators) == 0 or len(other.var_generators) == 0:
            raise ValueError("There is no generator variables in zonotope domain.")

        self.center -= other.center
        self.coef_generators = (jnp.array(self.coef_generators) - jnp.array(other.coef_generators)).tolist()
        # * self.var_generators remains the same
        self.concrete_domain.lower_bound = self.center - jnp.sum(jnp.abs(jnp.array(self.coef_generators)))
        self.concrete_domain.uppper_bound = self.center + jnp.sum(jnp.abs(jnp.array(self.coef_generators)))
        self.lower_bound = self.concrete_domain.lower_bound
        self.upper_bound = self.concrete_domain.upper_bound

        return


    def multiply(self, other: Any) -> Any:
        """
        * Zonotope multiplication, we don't need this function for now.
        """
        pass


    def meet(self, other: Any) -> Any:
        """
        ? how to implement meet operation for zonotope domain?
        """
        return 
    

    def join(self, other: Any) -> Any:
        result: ZonotopeDomain = ZonotopeDomain()
        
        # * for center point
        temp_box: BoxDomain = self.concrete_domain.join(other.concrete_domain)
        result.center = (temp_box.lower_bound + temp_box.upper_bound) / 2.0

        # * for old generator variables
        result.var_generators = self.var_generators
        for x, y in zip(self.coef_generators, other.coef_generators):
            min_value: float = min(x, y)
            max_value: float = max(x, y)
            is_included_zero: bool = True if min_value <= 0 and max_value >= 0 else False
            new_coef: float = 0 if is_included_zero else min(abs(min_value), abs(max_value))
            result.coef_generators.append(new_coef)
        
        # * for new generator variable
        result.var_generators.append(f"epsilon_{len(result.var_generators)}")
        # new_coef: float = temp_box.upper_bound - result.center - sum(map(abs, result.coef_generators))                            # * normal version
        new_coef: float = temp_box.upper_bound - result.center - float(jnp.sum(vmap(jnp.abs)(jnp.array(result.coef_generators))))   # * jax version
        result.coef_generators.append(new_coef)


        return result
    

if __name__ == "__main__":
    # * test ZonotopeDomain
    zonotope_domain1: ZonotopeDomain = ZonotopeDomain.from_lower_and_upper_bound(_lower_bound=-5.0, _upper_bound=1.0)
    zonotope_domain2: ZonotopeDomain = ZonotopeDomain.from_lower_and_upper_bound(_lower_bound=1.0, _upper_bound=2.0)

    print("zonotope_domain1: ", zonotope_domain1.lower_bound, zonotope_domain1.upper_bound, zonotope_domain1.center)
    print("zonotope_domain2: ", zonotope_domain2.lower_bound, zonotope_domain2.upper_bound, zonotope_domain2.center)
    print("---------------------------------------------------------------")
    print("zonotope concrete domain1: ", zonotope_domain1.concrete_domain.lower_bound, zonotope_domain1.concrete_domain.upper_bound)
    print("zonotope concrete domain2: ", zonotope_domain2.concrete_domain.lower_bound, zonotope_domain2.concrete_domain.upper_bound)
    print("---------------------------------------------------------------")

    zonotope_domain1.abstract_transformer()
    print("zonotope_domain1 after transformer: ", zonotope_domain1.lower_bound, zonotope_domain1.upper_bound, zonotope_domain1.center)
    print("name: ", zonotope_domain1.name)
    print("id: ", zonotope_domain1.id)
    print("var_generators: ", zonotope_domain1.var_generators)
    print("coef_generators: ", zonotope_domain1.coef_generators)
    print("---------------------------------------------------------------")

    zonotope_domain2.abstract_transformer()
    print("zonotope_domain2 after transformer: ", zonotope_domain2.lower_bound, zonotope_domain2.upper_bound, zonotope_domain2.center)
    print("name: ", zonotope_domain2.name)
    print("id: ", zonotope_domain2.id)
    print("var_generators: ", zonotope_domain2.var_generators)
    print("coef_generators: ", zonotope_domain2.coef_generators)
    print("---------------------------------------------------------------")

    # print("ADD:")
    # zonotope_domain1.add(zonotope_domain2)
    # print("zonotope_domain1 after add: ", zonotope_domain1.lower_bound, zonotope_domain1.upper_bound, zonotope_domain1.center)
    # print("name: ", zonotope_domain1.name)
    # print("id: ", zonotope_domain1.id)
    # print("center: ", zonotope_domain1.center)
    # print("var_generators: ", zonotope_domain1.var_generators)
    # print("coef_generators: ", zonotope_domain1.coef_generators)
    # print("---------------------------------------------------------------")

    # print("SUBSTRACT:")
    # zonotope_domain1.substract(zonotope_domain2)
    # print("zonotope_domain1 after substract: ", zonotope_domain1.lower_bound, zonotope_domain1.upper_bound, zonotope_domain1.center)
    # print("name: ", zonotope_domain1.name)
    # print("id: ", zonotope_domain1.id)
    # print("center: ", zonotope_domain1.center)
    # print("var_generators: ", zonotope_domain1.var_generators)
    # print("coef_generators: ", zonotope_domain1.coef_generators)
    # print("---------------------------------------------------------------")

    print("JOIN:")
    zonotope_domain3: ZonotopeDomain = zonotope_domain1.join(zonotope_domain2)
    print("zonotope_domain3 after join: ", zonotope_domain3.lower_bound, zonotope_domain3.upper_bound, zonotope_domain3.center)
    print("name: ", zonotope_domain3.name)
    print("id: ", zonotope_domain3.id)
    print("center: ", zonotope_domain3.center)
    print("var_generators: ", zonotope_domain3.var_generators)
    print("coef_generators: ", zonotope_domain3.coef_generators)
    print("---------------------------------------------------------------")
