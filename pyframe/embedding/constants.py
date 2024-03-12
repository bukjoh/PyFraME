from typing import Optional
from pyframe.embedding import polytensor, tensor_tools

from pyframe.embedding import engine


class Constants:
    """A Constants object represents a collection of variables.
    """
    initialized = False
    max_order = None
    interaction_tensor_template = None
    tensor_coefficients = None
    factorials = None
    double_factorials = None
    binomials = None
    trinomials = None
    degeneracies = None

    def __init__(self,
                 max_order: Optional[int] = 42,
                 t_rank: Optional[int] = 15
                 ):
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if t_rank < 0:
            raise ValueError("t_rank must be non-negative")

        if not self.initialized:
            self.max_order = max_order
            t_int_obj = polytensor.SecondDegreePolytensor(rank_1=t_rank, rank_2=t_rank, data_type=object)
            t_int_obj.write_interaction_tensor_multi_indices()
            t_pot_obj = polytensor.SecondDegreePolytensor(rank_1=t_rank, rank_2=t_rank, data_type=object)
            t_pot_obj.write_potential_tensor_multi_indices()
            self.interaction_tensor_template = t_int_obj.data
            self.potential_tensor_template = t_pot_obj.data
            self.tensor_coefficients = tensor_tools.compute_tensor_coefficients(max_order)
            self.factorials = tensor_tools.compute_factorials(max_order)
            self.double_factorials = tensor_tools.compute_double_factorials(max_order)
            self.binomials = tensor_tools.compute_binomial_coefficients(3 * max_order, 3 * max_order)
            self.trinomials = tensor_tools.compute_trinomial_coefficients(max_order, max_order, max_order,
                                                                          self.binomials)
            self.degeneracies = polytensor.FirstDegreePolytensor(self.max_order)
            for i in range(0, self.max_order + 1):
                self.degeneracies.write_to_data_block_wise(tensor_tools.compute_degeneracy_tensor(i, self.trinomials))

            engine.set_tensor_coefficients(self.tensor_coefficients,
                                           self.interaction_tensor_template,
                                           self.potential_tensor_template,
                                           t_rank,
                                           max_order)

            self.initialized = True


# TODO possible input of max order und t rank
values = Constants()
