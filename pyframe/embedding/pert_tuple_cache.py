"""Perturbation Tuple Cache structure from PyOpenRSP"""

import copy


# Perturbation class: Represented by operator oper (any hashable type) and associated frequency freq (float)
class rspPert:

    def __init__(self, oper, freq):
        # TODO: Add checks to see if o and f are valid types (o must be hashable, f must be scalar)

        # Operator label
        self.o = copy.deepcopy(oper)
        # Frequency
        self.f = copy.deepcopy(freq)

        # Hash
        self.h = hash((self.o, self.f))

    # Set IDs
    def setId(self, i):
        self.id = i

    # Remove IDs
    def rmId(self):
        self.id = None

    def __repr__(self):
        return f'rspPert obj {self.o}'


# Perturbation tuple class
class rspPertTuple:

    # For now just one freq cfg per tuple, could this later either by duplication or extension of the instance attributes
    # perts is list of rspPert instances
    def __init__(self, perts):

        # FIXME: Could need sorting method but for now not implemented

        # Make immutable
        self.p = tuple(copy.deepcopy(perts))

        # Hash of tuple of individual perturbation hashes
        self.h = hash(tuple([m.h for m in self.p]))

        # Determine frequency sum
        try:
            self.freqsum = sum([j.f for j in self.p])

        except TypeError:
            self.freqsum = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        self.index += 1
        try:
            return self.p[self.index]
        except IndexError:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return len(self.p)

    def __repr__(self):
        return f'rspPertTuple obj {self.p}'

    # Set perturbation IDs
    def setIds(self, st):

        for j in range(len(self.p)):
            self.p[j].setId(st + j)

    # Remove perturbation IDs
    def rmId(self):

        for j in range(len(self.p)):
            self.p[j].rmId()


# Property or contribution cache
# Assumes all individual pert tuples are sorted but not necessarily the tuple of tuples
class rspCache:

    # p_tuples is a tuple of perturbation tuples
    # k is an optional k rule choice
    # comps are (a set of) associated component tuples
    # values are associated values
    def __init__(self, p_tuples, k=None, n=None, comps=set(), values=None):

        # Begin with sorting
        self.p_tuples = self.sortTuples(p_tuples)

        # k rule choice
        self.k = k

        n_was_set = False

        # The n rule parameter can be determined if k is defined
        if k is not None:
            if n is None:
                if len(self.p_tuples) == 1:
                    self.n = len(self.p_tuples[0].p) - self.k - 1
                    n_was_set = True
        else:
            self.n = None

        if not n_was_set:
            if n is not None:
                self.n = n

        # Hash of (tuple of inner tuple hash and outer tuples hash hash, k rule choice if any)
        self.h = hash((tuple([m.h for m in self.p_tuples]), self.k))

        # Components
        self.comps = comps

        if values is not None:

            # Values (if specified) should be a dictionary of (component : value) pairs
            self.vals = values
            self.values_are_set = True

        # If values were not specified, initialize an empty dictionary and flag as not set
        else:

            self.vals = {}
            self.values_are_set = False

    # Set IDs of associated perturbation tuples
    def setIds(self):

        st = 0
        for j in range(len(self.p_tuples)):
            self.p_tuples[j].setIds(st)
            st += len(self.p_tuples[j].p)

    # Remove IDs
    def rmIds(self):

        for j in range(len(self.p_tuples)):
            self.p_tuples[j].rmId()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        self.index += 1
        try:
            return self.p_tuples[self.index]
        except IndexError:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return len(self.p_tuples)

    # FIXME: Currently not sorting, just returning input
    # May exempt first tuple from sorting
    def sortTuples(self, p_tuples):

        return copy.deepcopy(p_tuples)

    # Set values (the values must not already have keys in this cache instance)
    def setValues(self, values):

        # Walk through components
        for j in self.comps:
            assert (not (
                    j in self.vals)), 'When using rspCache.setValues(), the values must be unassigned for all components whose values are to be set'

            self.vals[j] = values[j]

        return

    # Add values (to already existing values)
    # Use callback if specified, otherwise use canonical addition
    def addValues(self, values, k1APk2B=None, host_data=None):

        for j in self.vals:

            # TODO: Maybe there could be an assertion here about the components of the values to be added already having values in this cache element

            # If an addition callback was specified, then use it
            if k1APk2B is not None:
                self.vals[j] = k1APk2B(1.0, self.vals[j], 1.0, values[j], host_data=host_data)

            # Otherwise use canonical addition
            else:
                self.vals[j] += values[j]

        return

    # Update self.comps to be the union of existing components and new components new_comps
    def compUnion(self, new_comps):

        self.comps = self.comps.union(new_comps)

        return

