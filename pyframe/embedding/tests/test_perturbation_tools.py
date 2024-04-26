"""Tests PyFraME.embedding.perturbatino_tools.py"""
from pyframe.embedding import perturbation_tools


def test_subsets_of_list():
    input_list = [1]
    assert len(perturbation_tools.subsets_of_list(input_list)) == 1
    assert perturbation_tools.subsets_of_list(input_list) == [[[1]]]
    input_list = [1, 2]
    assert len(perturbation_tools.subsets_of_list(input_list)) == 2
    assert perturbation_tools.subsets_of_list(input_list) == [[[2], [1]], [[1, 2]]]
    input_list = [1, 2, 3]
    assert len(perturbation_tools.subsets_of_list(input_list)) == 5
    assert perturbation_tools.subsets_of_list(input_list) == [[[1], [2, 3]],
                                                              [[1, 2, 3]],
                                                              [[3], [1, 2]],
                                                              [[3], [2], [1]],
                                                              [[2], [1, 3]]]
    input_list = [1, 2, 3, 4]
    assert len(perturbation_tools.subsets_of_list(input_list)) == 15
    assert perturbation_tools.subsets_of_list(input_list) == [[[3], [2, 4], [1]],
                                                              [[1], [2, 3], [4]],
                                                              [[3], [1, 2, 4]],
                                                              [[3], [2], [1, 4]],
                                                              [[3, 4], [2], [1]],
                                                              [[3], [2], [1], [4]],
                                                              [[2], [1, 3], [4]],
                                                              [[2], [1, 3, 4]],
                                                              [[1, 4], [2, 3]],
                                                              [[3, 4], [1, 2]],
                                                              [[2, 4], [1, 3]],
                                                              [[1, 2, 3, 4]],
                                                              [[1, 2, 3], [4]],
                                                              [[3], [4], [1, 2]],
                                                              [[1], [2, 3, 4]]]
    bell_numbers = [1, 1, 2, 5, 15, 52, 203]
    input_list = []
    for i in range(7):
        assert len(perturbation_tools.subsets_of_list(input_list)) == bell_numbers[i]
        input_list.append(i)
