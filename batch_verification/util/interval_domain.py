import sys
from typing import Any, Set

from abstract_interpretation import AbstractDomain


class BoxDomain(AbstractDomain):

    num_boxes: int = 0

    def __init__(
        self,
        _lower_bound: float = sys.float_info.min,
        _upper_bound: float = sys.float_info.max,
    ):
        super().__init__(_lower_bound=_lower_bound, _upper_bound=_upper_bound)
        self.name = f"box_{BoxDomain.num_boxes}"
        self.id = BoxDomain.num_boxes

        BoxDomain.num_boxes += 1

    def abstract_transformer(self):
        pass

    def add(self, other: Any) -> Any:
        self.lower_bound += other.lower_bound
        self.upper_bound += other.upper_bound

        return

    def substract(self, other: Any) -> Any:
        self.lower_bound -= other.lower_bound
        self.upper_bound -= other.upper_bound

        return

    def multiply(self, other: Any) -> Any:
        B: Set = {
            self.lower_bound * other.lower_bound,
            self.lower_bound * other.upper_bound,
            self.upper_bound * other.lower_bound,
            self.upper_bound * other.upper_bound,
        }

        self.lower_bound = min(B)
        self.upper_bound = max(B)

        return

    def meet(self, other: Any) -> Any:
        result: BoxDomain = BoxDomain()
        result.lower_bound = max(self.lower_bound, other.lower_bound)
        result.upper_bound = min(self.upper_bound, other.upper_bound)
        result.id = BoxDomain.num_boxes
        result.name = f"box_{BoxDomain.num_boxes}"
        BoxDomain.num_boxes += 1

        return result

    def join(self, other: Any) -> Any:
        result: BoxDomain = BoxDomain()
        result.lower_bound = min(self.lower_bound, other.lower_bound)
        result.upper_bound = max(self.upper_bound, other.upper_bound)
        result.id = BoxDomain.num_boxes
        result.name = f"box_{BoxDomain.num_boxes}"
        BoxDomain.num_boxes += 1

        return result


if __name__ == "__main__":
    # * test BoxDomain
    box_domain1: BoxDomain = BoxDomain.from_lower_and_upper_bound(
        _lower_bound=-5.0, _upper_bound=1.0
    )
    box_domain2: BoxDomain = BoxDomain.from_lower_and_upper_bound(
        _lower_bound=1.0, _upper_bound=2.0
    )
    print("name: ", box_domain1.name)
    print("name: ", box_domain2.name)
    print("---------------------------------------------------------------")
    print("id: ", box_domain1.id)
    print("id: ", box_domain2.id)

    box_domain1.add(other=box_domain2)
    print("ADD:")
    print("box_domain1.lower_bound: ", box_domain1.lower_bound)
    print("box_domain1.upper_bound: ", box_domain1.upper_bound)
    print("---------------------------------------------------------------")

    box_domain1.substract(other=box_domain2)
    print("SUBSTRACT:")
    print("box_domain1.lower_bound: ", box_domain1.lower_bound)
    print("box_domain1.upper_bound: ", box_domain1.upper_bound)
    print("---------------------------------------------------------------")

    box_domain1.multiply(other=box_domain2)
    print("MULTIPLY:")
    print("box_domain1.lower_bound: ", box_domain1.lower_bound)
    print("box_domain1.upper_bound: ", box_domain1.upper_bound)
    print("---------------------------------------------------------------")

    box_domain1.meet(other=box_domain2)
    print("MEET:")
    print("box_domain1.lower_bound: ", box_domain1.lower_bound)
    print("box_domain1.upper_bound: ", box_domain1.upper_bound)
    print("---------------------------------------------------------------")

    box_domain1.join(other=box_domain2)
    print("JOIN:")
    print("box_domain1.lower_bound: ", box_domain1.lower_bound)
    print("box_domain1.upper_bound: ", box_domain1.upper_bound)
    print("---------------------------------------------------------------")

    print("BoxDomain test finished.")
