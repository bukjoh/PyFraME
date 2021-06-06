# Copyright (C) 2017-2021  Jógvan Magnus Haugaard Olsen and Peter Reinholdt
#
# This file is part of PyFraME.
#
# PyFraME is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyFraME is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyFraME.  If not, see <https://www.gnu.org/licenses/>.
#

__all__ = ['PotentialDict', 'Potential']


class PotentialDict(dict):

    """Fragment potential dictionary"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Potential(object):

    """Container for fragment potentials"""

    def __init__(self, **kwargs):
        self._element = None
        self._coordinate = []
        self._M0 = []
        self._M1 = []
        self._M2 = []
        self._M3 = []
        self._M4 = []
        self._M5 = []
        self._M6 = []
        self._P00 = []
        self._P01 = []
        self._P02 = []
        self._P03 = []
        self._P11 = []
        self._P12 = []
        self._P13 = []
        self._P22 = []
        self._P23 = []
        self._P33 = []
        self._exclusion_list = []
        self._epsilon = None
        self._sigma = None
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                # TODO replace with exception
                exit('ERROR: unknown potential property "{0}"'.format(key))

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, element):
        assert isinstance(element, str)
        assert 1 <= len(element) <= 2
        self._element = element.title()

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, coordinate):
        assert all(isinstance(coord, float) for coord in coordinate)
        assert len(coordinate) == 3
        self._coordinate = coordinate

    @property
    def M0(self):
        return self._M0

    @M0.setter
    def M0(self, M0):
        if isinstance(M0, float):
            M0 = [M0]
        assert isinstance(M0, list)
        assert all(isinstance(component, float) for component in M0)
        assert len(M0) == 1
        self._M0 = M0

    @property
    def M1(self):
        return self._M1

    @M1.setter
    def M1(self, M1):
        assert isinstance(M1, list)
        assert all(isinstance(component, float) for component in M1)
        assert len(M1) == 3
        self._M1 = M1

    @property
    def M2(self):
        return self._M2

    @M2.setter
    def M2(self, M2):
        assert isinstance(M2, list)
        assert all(isinstance(component, float) for component in M2)
        assert len(M2) == 6
        self._M2 = M2

    @property
    def M3(self):
        return self._M3

    @M3.setter
    def M3(self, M3):
        assert isinstance(M3, list)
        assert all(isinstance(component, float) for component in M3)
        assert len(M3) == 10
        self._M3 = M3

    @property
    def M4(self):
        return self._M4

    @M4.setter
    def M4(self, M4):
        assert isinstance(M4, list)
        assert all(isinstance(component, float) for component in M4)
        assert len(M4) == 15
        self._M4 = M4

    @property
    def M5(self):
        return self._M5

    @M5.setter
    def M5(self, M5):
        assert isinstance(M5, list)
        assert all(isinstance(component, float) for component in M5)
        assert len(M5) == 21
        self._M5 = M5

    @property
    def M6(self):
        return self._M6

    @M6.setter
    def M6(self, M6):
        assert isinstance(M6, list)
        assert all(isinstance(component, float) for component in M6)
        assert len(M6) == 28
        self._M6 = M6

    @property
    def P00(self):
        return self._P00

    @P00.setter
    def P00(self, P00):
        assert isinstance(P00, list)
        assert all(isinstance(component, float) for component in P00)
        assert len(P00) == 1
        self._P00 = P00

    @property
    def P01(self):
        return self._P01

    @P01.setter
    def P01(self, P01):
        assert isinstance(P01, list)
        assert all(isinstance(component, float) for component in P01)
        assert len(P01) == 3
        self._P01 = P01

    @property
    def P02(self):
        return self._P02

    @P02.setter
    def P02(self, P02):
        assert isinstance(P02, list)
        assert all(isinstance(component, float) for component in P02)
        assert len(P02) == 6
        self._P02 = P02

    @property
    def P03(self):
        return self._P03

    @P03.setter
    def P03(self, P03):
        assert isinstance(P03, list)
        assert all(isinstance(component, float) for component in P03)
        assert len(P03) == 10
        self._P03 = P03

    @property
    def P11(self):
        return self._P11

    @P11.setter
    def P11(self, P11):
        assert isinstance(P11, list)
        assert all(isinstance(component, float) for component in P11)
        assert len(P11) == 6
        self._P11 = P11

    @property
    def P12(self):
        return self._P12

    @P12.setter
    def P12(self, P12):
        assert isinstance(P12, list)
        assert all(isinstance(component, float) for component in P12)
        assert len(P12) == 10
        self._P12 = P12

    @property
    def P13(self):
        return self._P13

    @P13.setter
    def P13(self, P13):
        assert isinstance(P13, list)
        assert all(isinstance(component, float) for component in P13)
        assert len(P13) == 15
        self._P13 = P13

    @property
    def P22(self):
        return self._P22

    @P22.setter
    def P22(self, P22):
        assert isinstance(P22, list)
        assert all(isinstance(component, float) for component in P22)
        assert len(P22) == 15
        self._P22 = P22

    @property
    def P23(self):
        return self._P23

    @P23.setter
    def P23(self, P23):
        assert isinstance(P23, list)
        assert all(isinstance(component, float) for component in P23)
        assert len(P23) == 21
        self._P23 = P23

    @property
    def P33(self):
        return self._P33

    @P33.setter
    def P33(self, P33):
        assert isinstance(P33, list)
        assert all(isinstance(component, float) for component in P33)
        assert len(P33) == 28
        self._P33 = P33

    @property
    def exclusion_list(self):
        return self._exclusion_list

    @exclusion_list.setter
    def exclusion_list(self, exclusion_list):
        assert isinstance(exclusion_list, list)
        assert all(isinstance(component, int) for component in exclusion_list)
        self._exclusion_list = exclusion_list

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        assert isinstance(epsilon, float)
        self._epsilon = epsilon

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        assert isinstance(sigma, float)
        self._sigma = sigma
