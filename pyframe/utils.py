# coding=utf-8
"""Blablabla"""

from numba import jit, float64
import numpy as np

__all__ = ['BOHR2AA', 'AA2BOHR', 'element2radius', 'element2charge', 'element2mass', 'compute_angle',
           'compute_distance', 'compute_distance_matrix', 'get_minimum_distance',
           'get_bond_length', 'scale_bond_length']

BOHR2AA = 0.5291772108
AA2BOHR = 1.0 / BOHR2AA

elements = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
            'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
            'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
            'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
            'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt')

masses = (1.007825, 4.002603, 7.016005, 9.012183, 11.009305, 12.000000, 14.003074, 15.994915,
          18.998403, 19.992439, 22.989770, 23.985045, 26.981541, 27.976928, 30.973763, 31.972072,
          34.968853, 39.962383, 38.963708, 39.962591, 44.955914, 47.947947, 50.943963, 51.940510,
          54.938046, 55.934939, 58.933198, 57.935347, 62.929599, 63.929145, 68.925581, 73.921179,
          74.921596, 79.916521, 78.918336, 83.911506, 84.911800, 87.905625, 88.905856, 89.904708,
          92.906378, 97.905405, 97.907216, 101.904348, 102.905503, 105.903475, 106.905095,
          113.903361, 114.903875, 119.902199, 120.903824, 129.906229, 126.904477, 131.904148,
          132.905429, 137.905232, 138.906347, 139.905433, 140.907647, 141.907719, 144.912743,
          151.919728, 152.921225, 157.924019, 158.925342, 163.929171, 164.930319, 165.930290,
          168.934212, 173.938859, 174.940770, 179.946546, 180.947992, 183.950928, 186.955744,
          191.961467, 192.962917, 194.964766, 196.966543, 201.970617, 204.974401, 207.976627,
          208.980374, 208.982404, 209.987126, 222.017571, 223.019733, 226.025403, 227.027750,
          232.038051, 231.035880, 238.050785, 237.048168, 244.064199, 243.061373, 247.070347,
          247.070300, 251.079580, 252.082944, 257.095099, 258.098570, 259.100931, 260.105320,
          261.108690, 262.113760, 263.118220, 262.122930, 269.134100, 267.138000)

# Only single bonds. The first 96 are from DOI: 10.1039/B801115J,
# and the rest is from DOI: 10.1002/chem.200800987.
# Now using ionic radii for typical mono- and divalent cations, e.g. Li, Na, Ca, etc.,
# from DOI: 10.1107/S0567739476001551
element2radius = {'H': 0.31, 'He': 0.28, 'Li': 0.90, 'Be': 0.56, 'B': 0.84, 'C': 0.76, 'N': 0.71,
                  'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.16, 'Mg': 0.86, 'Al': 1.21, 'Si': 1.11,
                  'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 1.52, 'Ca': 1.14, 'Sc': 1.70,
                  'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50,
                  'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
                  'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90,
                  'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
                  'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39,
                  'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07,
                  'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98,
                  'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90,
                  'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51,
                  'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45,
                  'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 'Fr': 2.60,
                  'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96, 'Np': 1.90,
                  'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65,
                  'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61, 'Rf': 1.57, 'Db': 1.49,
                  'Sg': 1.43, 'Bh': 1.41, 'Hs': 1.34, 'Mt': 1.29}
# This is the dict using only covalent atomic radii.
# element2radius = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71,
#                  'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11,
#                  'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.70,
#                  'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50,
#                  'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
#                  'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90,
#                  'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
#                  'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39,
#                  'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07,
#                  'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98,
#                  'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90,
#                  'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51,
#                  'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45,
#                  'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 'Fr': 2.60,
#                  'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96, 'Np': 1.90,
#                  'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65,
#                  'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61, 'Rf': 1.57, 'Db': 1.49,
#                  'Sg': 1.43, 'Bh': 1.41, 'Hs': 1.34, 'Mt': 1.29}

# element2charge = {elem: charge + 1 for charge, elem in enumerate(elements)}
element2charge = {'Ru': 44, 'Re': 75, 'Rf': 104, 'Ra': 88, 'Rb': 37, 'Rn': 86,
                  'Rh': 45, 'Be': 4, 'Ba': 56, 'Bh': 107, 'Bi': 83, 'Bk': 97,
                  'Br': 35, 'H': 1, 'P': 15, 'Os': 76, 'Es': 99, 'Hg': 80,
                  'Ge': 32, 'Gd': 64, 'Ga': 31, 'Pr': 59, 'Pt': 78, 'Pu': 94,
                  'C': 6, 'Pb': 82, 'Pa': 91, 'Pd': 46, 'Cd': 48, 'Po': 84,
                  'Pm': 61, 'Hs': 108, 'Ho': 67, 'Hf': 72, 'K': 19, 'He': 2,
                  'Md': 101, 'Mg': 12, 'Mo': 42, 'Mn': 25, 'O': 8, 'Mt': 109,
                  'S': 16, 'W': 74, 'Zn': 30, 'Eu': 63, 'Zr': 40, 'Er': 68,
                  'Ni': 28, 'No': 102, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10,
                  'Np': 93, 'Fr': 87, 'Fe': 26, 'Fm': 100, 'B': 5, 'F': 9,
                  'Sr': 38, 'N': 7, 'Kr': 36, 'Si': 14, 'Sn': 50, 'Sm': 62,
                  'V': 23, 'Sc': 21, 'Sb': 51, 'Sg': 106, 'Se': 34, 'Co': 27,
                  'Cm': 96, 'Cl': 17, 'Ca': 20, 'Cf': 98, 'Ce': 58, 'Xe': 54,
                  'Lu': 71, 'Cs': 55, 'Cr': 24, 'Cu': 29, 'La': 57, 'Li': 3,
                  'Tl': 81, 'Tm': 69, 'Lr': 103, 'Th': 90, 'Ti': 22, 'Te': 52,
                  'Tb': 65, 'Tc': 43, 'Ta': 73, 'Yb': 70, 'Db': 105, 'Dy': 66,
                  'I': 53, 'U': 92, 'Y': 39, 'Ac': 89, 'Ag': 47, 'Ir': 77,
                  'Am': 95, 'Al': 13, 'As': 33, 'Ar': 18, 'Au': 79, 'At': 85,
                  'In': 49}

# element2mass = {elem: mass for elem, mass in zip(elements, masses)}
element2mass = {'Ru': 101.904348, 'Re': 186.955744, 'Rf': 261.10869, 'Ra': 226.025403,
                'Rb': 84.9118, 'Rn': 222.017571, 'Rh': 102.905503, 'Be': 9.012183,
                'Ba': 137.905232, 'Bh': 262.12293, 'Bi': 208.980374, 'Bk': 247.0703,
                'Br': 78.918336, 'H': 1.007825, 'P': 30.973763, 'Os': 191.961467, 'Es': 252.082944,
                'Hg': 201.970617, 'Ge': 73.921179, 'Gd': 157.924019, 'Ga': 68.925581,
                'Pr': 140.907647, 'Pt': 194.964766, 'Pu': 244.064199, 'C': 12.0,
                'Pb': 207.976627, 'Pa': 231.03588, 'Pd': 105.903475, 'Cd': 113.903361,
                'Po': 208.982404, 'Pm': 144.912743, 'Hs': 269.1341, 'Ho': 164.930319,
                'Hf': 179.946546, 'K': 38.963708, 'He': 4.002603, 'Md': 258.09857,
                'Mg': 23.985045, 'Mo': 97.905405, 'Mn': 54.938046, 'O': 15.994915,
                'Mt': 267.138, 'S': 31.972072, 'W': 183.950928, 'Zn': 63.929145, 'Eu': 152.921225,
                'Zr': 89.904708, 'Er': 165.93029, 'Ni': 57.935347, 'No': 259.100931,
                'Na': 22.98977, 'Nb': 92.906378, 'Nd': 141.907719, 'Ne': 19.992439,
                'Np': 237.048168, 'Fr': 223.019733, 'Fe': 55.934939, 'Fm': 257.095099,
                'B': 11.009305, 'F': 18.998403, 'Sr': 87.905625, 'N': 14.003074,
                'Kr': 83.911506, 'Si': 27.976928, 'Sn': 119.902199, 'Sm': 151.919728,
                'V': 50.943963, 'Sc': 44.955914, 'Sb': 120.903824, 'Sg': 263.11822,
                'Se': 79.916521, 'Co': 58.933198, 'Cm': 247.070347, 'Cl': 34.968853,
                'Ca': 39.962591, 'Cf': 251.07958, 'Ce': 139.905433, 'Xe': 131.904148,
                'Lu': 174.94077, 'Cs': 132.905429, 'Cr': 51.94051, 'Cu': 62.929599,
                'La': 138.906347, 'Li': 7.016005, 'Tl': 204.974401, 'Tm': 168.934212,
                'Lr': 260.10532, 'Th': 232.038051, 'Ti': 47.947947, 'Te': 129.906229,
                'Tb': 158.925342, 'Tc': 97.907216, 'Ta': 180.947992, 'Yb': 173.938859,
                'Db': 262.11376, 'Dy': 163.929171, 'I': 126.904477, 'U': 238.050785,
                'Y': 88.905856, 'Ac': 227.02775, 'Ag': 106.905095, 'Ir': 192.962917,
                'Am': 243.061373, 'Al': 26.981541, 'As': 74.921596, 'Ar': 39.962383,
                'Au': 196.966543, 'At': 209.987126, 'In': 114.903875}


@jit(float64(float64[:], float64[:], float64[:]))
def compute_angle(a, b, c):
    """Compute angle between points a, b and c"""
    ab = compute_distance(a, b)
    ac = compute_distance(a, c)
    bc = compute_distance(b, c)
    return np.arccos((ab**2 - ac**2 + bc**2) / (2.0 * ab * bc))


@jit(float64(float64[:], float64[:]))
def compute_distance(a, b):
    """Compute distance between point a and b"""
    assert len(a) == len(b)
    distance = 0.0
    for i in range(len(a)):
        distance += (b[i] - a[i])**2
    return np.sqrt(distance)


@jit(float64[:,:](float64[:,:], float64[:,:]))
def compute_distance_matrix(A, B):
    """Compute distance matrix between matrices A nd B"""
    distance_matrix = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            distance_matrix[i,j] = compute_distance(A[i], B[j])
    return distance_matrix


def get_minimum_distance(first_fragment, second_fragment):
    """Calculates minimum atom-atom distance between two fragments"""
    distances = compute_distance_matrix(first_fragment.coordinate_matrix, second_fragment.coordinate_matrix)
    min_dist = distances.min()
    return min_dist


def get_bond_length(a, b):
    return element2radius[a] + element2radius[b]


def scale_bond_length(acceptor_atom, donor_atom):
    difference_coordinate = acceptor_atom.coordinate - donor_atom.coordinate
    factor = get_bond_length(acceptor_atom.element, donor_atom.element)
    factor /= np.linalg.norm(difference_coordinate)
    return acceptor_atom.coordinate - factor * difference_coordinate
