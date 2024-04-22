from __future__ import annotations

import numpy as np
import copy

from typing import Optional
from pyframe.embedding import subsystem, tensor_tools, constants
from pyframe.embedding.pert_tuple_cache import rspPertTuple, rspCache


def comps_for_these_p_tuples(p_tuples: tuple,
                             comps: set,
                             id_map: Optional[list] = None,
                             mirror: Optional[bool] = False,
                             return_comp_map: Optional[bool] = False):
    """
    Args:
        p_tuples:
        comps:
        as_dict:
        id_map:
        mirror:
        return_comp_map:

    Returns:
        new_comps: Transformed components
        comp_mapping: Mapping from original components to transformed components
    """
    new_comps = set()
    comp_mapping = {}
    # Go through the original components and use the perturbation ID attribute
    # of the p_tuples perturbations to determine the corresponding form of that
    # component for the perturbations on the p_tuples form
    if mirror is False:
        for i in comps:
            new_comp = []
            for j in p_tuples:
                if id_map:
                    new_comp.append(tuple([i[0][id_map[1][id_map[0].index(k.id)]] for k in j]))
                else:
                    new_comp.append(tuple([i[0][k.id] for k in j]))
            new_comps.add(tuple(copy.deepcopy(new_comp)))
            if return_comp_map:
                comp_mapping[i] = tuple(copy.deepcopy(new_comp))
    if mirror is True:
        for i in comps:
            new_comp = []
            for j in p_tuples:
                if id_map:
                    new_comp.append(tuple([i[1][id_map[1][id_map[0].index(k.id)]] for k in j]))
                else:
                    new_comp.append(tuple([i[1][k.id] for k in j]))
            new_comps.add(tuple(copy.deepcopy(new_comp)))
            # Update the mapping dictionary
            if return_comp_map:
                comp_mapping[i] = tuple(copy.deepcopy(new_comp))
    if return_comp_map:
        return comp_mapping
    else:
        return new_comps


def kn_skip(perts, kn):
    if 0 in [i.id for i in perts.p]:
        if len(perts.p) > kn[0]:
            return True
        return False

    return n_skip(perts, kn[1])


def n_skip(perts, n):
    if len(perts.p) > n:
        return True
    return False


def ind_contr_recurse(perts: rspPertTuple,
                      res_tuples: list,
                      fbf_cache: dict,
                      comps: set,
                      k: int,
                      n: int,
                      unique_f_cache: Optional[dict] = None,
                      dryrun: Optional[bool] = True,
                      e_ind_contr: Optional[dict] = None,
                      debug_print: Optional[bool] = False
                      ) -> None:
    """Recursion to identify energy-type contributions for a perturbation tuple.

    Args:
        perts: Perturbation tuple.
        res_tuples:
        fbf_cache: The cache that will be filled with the identified terms during the dryrun or the source of the calculated
        data if non-dryrun.
        unique_f_cache:
        k:
        n:
        comps:
        dryrun: Flag that indicates if only the Terms are identified, or if results are collected or calculated.
        e_ind_contr: Cache of induced contributions where results are collected in.
        debug_print:
    """
    # Recurse with product rule
    if len(perts.p) > 0:
        # Take out the first pert
        new_p_tuple = list(copy.deepcopy(perts.p))
        first_pert = copy.deepcopy(new_p_tuple[0])
        del new_p_tuple[0]
        new_perts = rspPertTuple(new_p_tuple)
        # Product rule
        for i in range(len(res_tuples)):
            new_res_tuples = copy.deepcopy(res_tuples)
            new_res_tuples[i].append(first_pert)
            ind_contr_recurse(perts=new_perts,
                              res_tuples=new_res_tuples,
                              fbf_cache=fbf_cache,
                              k=k,
                              n=n,
                              comps=comps,
                              unique_f_cache=unique_f_cache,
                              dryrun=dryrun,
                              e_ind_contr=e_ind_contr,
                              debug_print=debug_print)
    # End of recursion
    else:
        if debug_print:
            print("\nThe end of the recursion is reached.")
        # After recursion section, crystallize res_tuples into a p_tuple tuple
        for i in range(len(res_tuples)):
            res_tuples[i] = rspPertTuple(res_tuples[i])
        new_p_tuples = tuple(copy.deepcopy(res_tuples))
        if dryrun is True:
            # Make cache instance
            # unperturbed case
            if len(res_tuples[0]) == 0 and len(res_tuples[1]) == 0:
                new_comps = comps
            else:
                # Identify requested components for this tuple
                new_comps = comps_for_these_p_tuples(tuple([i.p for i in res_tuples]), set([i for i in comps]))
            new_inst = rspCache(p_tuples=new_p_tuples, comps=new_comps, k=k, n=n)
            if debug_print is True:
                print("Create new instance..",
                      "\nWith the Perturbation Tuple is ", [[j.o for j in i.p] for i in new_p_tuples],
                      "\nWith the ids", [[j.id for j in i.p] for i in new_p_tuples],
                      " and the components", new_inst.comps)
            # See if there are any cache elements corresponding to the instance
            if debug_print is True:
                print("\nFor dryrun:",
                      "\nCheck if PertTuple ids of each F is unique.")
                # "\nCheck if id's are unique.")
            for m, p_tup in enumerate(new_inst.p_tuples):
                individual_F_comps = set([i[m] for i in new_inst.comps])
                if debug_print is True:
                    print("\n" + str([j.o for j in p_tup.p]) + " with the components " + str(individual_F_comps))
                # check for unique ids
                if hash(tuple([j.id for j in p_tup.p])) in unique_f_cache.keys():
                    unique_f_cache[hash(tuple([j.id for j in p_tup.p]))].compUnion(individual_F_comps)
                    if debug_print is True:
                        print("\nNot unique F, therefore form component Union.")
                else:
                    individual_F = rspCache(p_tuples=[p_tup], comps=individual_F_comps, k=k, n=n)
                    unique_f_cache[hash(tuple([j.id for j in p_tup.p]))] = individual_F
                    if debug_print is True:
                        print("\nUnique F, therefore create new instance and add to Cache.")
                        print("Create new instance..",
                              "\nWith the Perturbation Tuple is ", [[j.o for j in i.p] for i in individual_F.p_tuples],
                              "\nWith the ids", [[j.id for j in i.p] for i in individual_F.p_tuples],
                              " and the components", individual_F.comps)
            if debug_print is True:
                print("\nFor non-dryrun.",
                      "\nAll Terms are appended.")
            fbf_cache[hash(tuple([tuple([j.id for j in m]) for m in res_tuples]))] = new_inst
        else:
            # unperturbed case
            comp_map = None
            if len(res_tuples[0]) == 0 and len(res_tuples[1]) == 0:
                for comp in comps:
                    comp_map = {comp: comp}
            else:
                # Identify requested components for this tuple
                comp_map = comps_for_these_p_tuples(p_tuples=tuple([i.p for i in res_tuples]),
                                                    comps=set([i for i in comps]),
                                                    return_comp_map=True)
            res_tuple_hash = hash(tuple([tuple([j.id for j in m]) for m in res_tuples]))
            for comp in comps:
                e_ind_contr[comp] += -0.5 * fbf_cache[res_tuple_hash].vals[comp_map[comp]]


def rsp_field_recurse(perts: rspPertTuple,
                      res_tuples: list,
                      comps: set,
                      cache: dict,
                      kn: list,
                      id_map: list,
                      dryrun: Optional[bool] = True,
                      vals_to_set: Optional[rspCache] = None,
                      debug_print: Optional[bool] = False
                      ) -> None:
    """Recursion to identify energy-type contributions for a perturbation tuple.

    Args:
        perts: Perturbation tuple.
        res_tuples:
        comps: Components of the original tuple (what is the original tuple?) to be calculated.
        dryrun: Flag indicating if the recursion is supposed to identify terms (True) or distribute terms (False).
        cache: The cache that will be filled with the identified terms during the dryrun or the source of the calculated
        data if non-dryrun.
        kn:
        id_map:
        dryrun:
        vals_to_set:
        debug_print:

    Returns:

    """
    if len(perts.p) > 0:

        # Take out the first pert
        new_p_tuple = list(copy.deepcopy(perts.p))
        first_pert = copy.deepcopy(new_p_tuple[0])
        del new_p_tuple[0]
        new_perts = rspPertTuple(new_p_tuple)
        # Product rule

        for i in range(len(res_tuples)):
            new_res_tuples = copy.deepcopy(res_tuples)
            new_res_tuples[i].append(first_pert)

            rsp_field_recurse(new_perts, new_res_tuples, comps, cache, kn, id_map, dryrun, vals_to_set, debug_print)

        # Chain rule
        if len(res_tuples) < 2:
            new_res_tuples = copy.deepcopy(res_tuples)
            new_res_tuples.append([])
            new_res_tuples[len(new_res_tuples) - 1].append(first_pert)

            rsp_field_recurse(new_perts, new_res_tuples, comps, cache, kn, id_map, dryrun, vals_to_set, debug_print)
    # End of recursion
    else:
        if debug_print:
            print("\n-----------------------------------------------------------------------------")
            print("\nThe end of the recursion is reached.")
        # unperturbed case
        if len(res_tuples) == 1:
            if len(res_tuples[0]) == 0:
                res_tuples = [rspPertTuple([])]

        # After recursion section, crystallize res_tuples into a p_tuple tuple
        if len(res_tuples) == 1:
            res_tuples.append([])
        for i in range(len(res_tuples)):
            res_tuples[i] = rspPertTuple(res_tuples[i])
        new_p_tuples = tuple(copy.deepcopy(res_tuples))
        # Test for kn rule truncation
        skip_by_kn = False

        for i in range(1, len(new_p_tuples)):
            # for field only F^1 is possible, so there is only one D!
            # -> len(new_p_tuples) is always 2. (one for F one for D)
            skip_by_kn = kn_skip(new_p_tuples[i], kn)  # or skip_by_kn

        # Skip
        skip_el = False
        if not skip_by_kn:
            for j in new_p_tuples[0].p:
                if j.o == 'EL':
                    skip_el = True

        if debug_print is True:
            print("Check for kn rule truncation for k=" + str(kn[0]) + " n=" + str(kn[1]) + ".. " + str(skip_by_kn))
            print("\nCheck if first Term is 'EL'.. " + str(skip_el))
            if skip_by_kn or skip_el:
                print("\nThe Term" + str([[j.o for j in i.p] for i in new_p_tuples]) + " is skipped.")
            else:
                print("\nThe Term" + str([[j.o for j in i.p] for i in new_p_tuples]) + " is NOT skipped.")

        # If not truncating
        if not (skip_by_kn or skip_el):
            # set new comps
            if res_tuples[0]:
                set_list = []
                for comp in comps:
                    set_list.append(tuple([comp, ()]))
                new_comps = comps_for_these_p_tuples(p_tuples=tuple([i.p for i in res_tuples]),
                                                     comps=set(set_list),
                                                     id_map=id_map)
            elif not res_tuples[0] and not res_tuples[1]:
                new_comps = {tuple([(), ()])}
            else:
                set_list = []
                for comp in comps:
                    set_list.append(tuple([(), comp]))

                new_comps = comps_for_these_p_tuples(p_tuples=tuple([i.p for i in res_tuples]),
                                                     comps=set(set_list),
                                                     id_map=id_map,
                                                     mirror=True)

            # Make cache instance -> one cache per term with necessary components
            new_inst = rspCache(new_p_tuples, comps=new_comps)

            if debug_print is True:
                print("\nCreate new instance..",
                      "\nWith the Perturbation Tuple is ", [[j.o for j in i.p] for i in new_inst.p_tuples],
                      "\nWith the ids", [[j.id for j in i.p] for i in new_inst.p_tuples],
                      " and the components", new_inst.comps)

            if dryrun is True:
                if debug_print is True:
                    print("\nCheck if first perturbation tuple " + str([j.o for j in new_inst.p_tuples[0].p])
                          + " hash already in cache..")
                if new_inst.p_tuples[0].h in cache.keys():
                    if debug_print is True:
                        print("First perturbation tuple hash IS in cache.",
                              "Check p_tuples are unique")
                    # check list for unique ids
                    h = [cache_element.h for cache_element in cache[new_inst.p_tuples[0].h]]
                    # h = [hash(tuple([tuple([pert.id for pert in p_tuple.p]) for p_tuple in cache_element.p_tuples]))
                    #     for cache_element in cache[new_inst.p_tuples[0].h]]
                    try:
                        # cache_where = h.index(hash(tuple([tuple([pert.id for pert in p_tuple.p])
                        #                                  for p_tuple in new_inst.p_tuples])))
                        cache_where = h.index(new_inst.h)
                        # dry_cache[p_tup.h].compUnion(individual_F_comps)
                        if debug_print is True:
                            # print("Ids are not unique.")
                            print("\np_tuples are not unique.")
                    except ValueError:
                        cache_where = -1
                        if debug_print is True:
                            print("\np_tuples are unique.")

                # Otherwise the element was obviously not in cache, so also make a new cache segment for this order
                else:
                    if debug_print is True:
                        print("First perturbation tuple hash is NOT in cache yet.")
                    cache_where = -1
                    cache[new_inst.p_tuples[0].h] = []

                # If not already in cache
                if cache_where == -1:
                    # Add this prop instance to the cache
                    cache[new_inst.p_tuples[0].h].append(copy.deepcopy(new_inst))
                    if debug_print is True:
                        print("Term will be appended.")
                # If already in cache -> comp union?
                else:
                    cache[new_inst.p_tuples[0].h][cache_where].compUnion(new_inst.comps)
                    if debug_print is True:
                        print("Component Union is formed.")
                    # Make union of components identified here and components in existing cache entry
                    # cache[new_inst.p_tuples[0].h][h.index(hash(tuple([tuple([pert.id for pert in p_tuple.p])
                    #                                                  for p_tuple in new_inst.p_tuples])))].compUnion(
                    #    new_comps)

            # NON DRYRUN
            else:
                # FIXME test the non-dryrun part.
                # values is a dictionary {comp1: unique_field1, ...} -> grab from FD cache
                if vals_to_set:
                    h = [cache_element.h for cache_element in cache[new_inst.p_tuples[0].h]]
                    cache_where = h.index(new_inst.h)
                    # find the corresponding rspCache instance
                    # map new_comps onto comps ?
                    if res_tuples[0]:
                        for unique_f_comp in comps:
                            new_comps = comps_for_these_p_tuples(p_tuples=tuple([i.p for i in res_tuples]),
                                                                 comps={tuple([unique_f_comp, ()])},
                                                                 id_map=id_map)
                            # FIXME I think this loop is unnecessary, since new_comps will actually be only one comp?
                            #  -> but dict of tuple so will just be tuple.. so should be ok
                            for new_comp in new_comps:
                                # transform comp format
                                if unique_f_comp in vals_to_set.vals:
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                                else:
                                    vals_to_set.vals[unique_f_comp] = np.zeros(
                                        cache[new_inst.p_tuples[0].h][cache_where].vals[new_comp].shape,
                                        dtype=np.float64)
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                    elif not res_tuples[0] and not res_tuples[1]:
                        new_comps = {tuple([(), ()])}
                        for unique_f_comp in comps:
                            for new_comp in new_comps:
                                if unique_f_comp in vals_to_set.vals:
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                                else:
                                    vals_to_set.vals[unique_f_comp] = np.zeros(
                                        cache[new_inst.p_tuples[0].h][cache_where].vals[new_comp].shape,
                                        dtype=np.float64)
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                    else:
                        for unique_f_comp in comps:
                            # TODO here mirror comes into play? what does it do again?
                            new_comps = comps_for_these_p_tuples(p_tuples=tuple([i.p for i in res_tuples]),
                                                                 comps={tuple([(), unique_f_comp])},
                                                                 id_map=id_map,
                                                                 mirror=True)
                            for new_comp in new_comps:
                                # transform comp format
                                if unique_f_comp in vals_to_set.vals:
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                                else:
                                    vals_to_set.vals[unique_f_comp] = np.zeros(
                                        cache[new_inst.p_tuples[0].h][cache_where].vals[new_comp].shape,
                                        dtype=np.float64)
                                    vals_to_set.vals[unique_f_comp] += cache[new_inst.p_tuples[0].h][cache_where].vals[
                                        new_comp]
                    vals_to_set.values_are_set = True
                else:
                    raise Exception("Non-dryrun without rspCache! Give rspCache for which vals are supposed to be set.")
    return


def calc_pert_el_fields(density_bank: dict,
                        f_el_cache: dict,
                        quantum_subsystem: subsystem.QuantumSubsystem,
                        coordinates: np.ndarray):
    """Calculates and sets the components of all electric Field derivative Terms of the f_el_cache.

    """
    td_cache = {}
    for key in f_el_cache.keys():
        #
        if len(f_el_cache[key]) > 1:
            first_iter = True
            for f_el in f_el_cache[key]:
                # skip unperturbed f_el
                if not f_el.p_tuples[0].p and not f_el.p_tuples[1].p:
                    continue
                # if f_el.p_tuples
                # check if hash p_tuples already in tD cache
                if f_el.h not in td_cache.keys():
                    td_cache[f_el.h] = {}

                # bonus from list ordering -> same t matrix for all of them

                # array of vals has to be in the same order as comps -> so loop over comps
                array_of_vals = np.zeros(len(f_el.comps), dtype=np.float64)
                for i, comp in enumerate(f_el.comps):
                    if hash(comp) in td_cache[f_el.h]:
                        array_of_vals[i] = td_cache[f_el.h][hash(comp)]
                        # use already existing tD value
                    else:
                        t_cache = {}
                        if first_iter:
                            # TODO maybe this is actually too much memory?
                            # calculate t and fill up the t_cache key=comp then contract with density
                            t_cache[comp] = None
                            # and contract
                            # electric_field[0] = np.einsum("ij, ij", density, ef_results.x_to_numpy())
                            # electric_field[1] = np.einsum("ij, ij", density, ef_results.y_to_numpy())
                            # electric_field[2] = np.einsum("ij, ij", density, ef_results.z_to_numpy())

                            density_to_use = density_bank[f_el.p_tuples[1].h]
                            # add result to tD cache td_cache[f_el.h][hash(comp)]
                            # and to array_of_vals[i]
                            td_cache[f_el.h][hash(comp)] = None
                            array_of_vals[i] = td_cache[f_el.h][hash(comp)]
                        else:
                            # check if comp in t_cache
                            if comp not in t_cache:
                                # calculate and add to t_cache
                                t_cache[comp] = None
                            # t_cache[comp] should now be set and usable!
                            density_to_use = density_bank[f_el.p_tuples[1].h]
                            # add result to tD cache td_cache[f_el.h][hash(comp)]
                            # and to array_of_vals[i]
                            td_cache[f_el.h][hash(comp)] = None
                            array_of_vals[i] = td_cache[f_el.h][hash(comp)]

                f_el_cache[key].setValues(values=array_of_vals)
                f_el_cache[key].values_are_set = True
                first_iter = False

        else:
            # no bonus from list ordering -> different t matrix for all of them
            for f_el in f_el_cache[key]:
                # check if hash p_tuples already in tD cache
                if f_el.h not in td_cache.keys():
                    td_cache[f_el.h] = {}
                # array of vals has to be in the same order as comps -> so loop over comps
                array_of_vals = np.zeros(len(f_el.comps), dtype=np.float64)
                for i, comp in enumerate(f_el.comps):
                    if hash(comp) in td_cache[f_el.h]:
                        array_of_vals[i] = td_cache[f_el.h][hash(comp)]
                        # use already existing tD value
                    else:
                        # calculate t and contract with D and add to array_of_vals[i] and tD cache
                        density_to_use = density_bank[f_el.p_tuples[1].h]
                        td_cache[f_el.h][hash(comp)] = None
                        array_of_vals[i] = td_cache[f_el.h][hash(comp)]
                f_el_cache[key].setValues(values=array_of_vals)
                f_el_cache[key].values_are_set = True


def calc_nuclei_fields(unique_f_cache,
                       quantum_subsystem: subsystem.QuantumSubsystem,
                       coordinates: np.ndarray):
    """Adds perturbed Fields from the Nuclei to the unique Field cache.

    """
    for key in unique_f_cache.keys():
        # Skips F unperturbed
        if not unique_f_cache[key].p_tuples[0].p:
            continue
        # Skip if any 'EL' perturbation
        skip_el = False
        for j in unique_f_cache[key].p_tuples[0].p:
            if j.o == 'EL':
                skip_el = True
        if skip_el:
            continue
        # Loop over components
        contr_added = False
        for comp in unique_f_cache[key].comps:
            pert_idx = None
            multi_index_x = [np.array([1, 0, 0], dtype=np.int64),
                             np.array([0, 0, 0], dtype=np.int64)]
            multi_index_y = [np.array([0, 1, 0], dtype=np.int64),
                             np.array([0, 0, 0], dtype=np.int64)]
            multi_index_z = [np.array([0, 0, 1], dtype=np.int64),
                             np.array([0, 0, 0], dtype=np.int64)]
            field_deriv = np.zeros(3, dtype=np.int64)
            nuc_index = None
            for pert in comp:
                # Single integer to describe index and multiindex
                nuc_index = pert // 3
                if pert_idx is None:
                    pert_idx = nuc_index
                if nuc_index != pert_idx:
                    field_deriv = np.zeros(3, dtype=np.int64)
                    break
                mod = pert % 3
                field_deriv[mod] += 1
            multi_index_x[0] += field_deriv
            multi_index_y[0] += field_deriv
            multi_index_z[0] += field_deriv
            # If there is no comp contribution from F_el yet
            if comp not in unique_f_cache[key].vals.keys():
                unique_f_cache[key].vals[comp] = np.zeros(3, dtype=np.float64)
            # calculate nuclear field of that multi index on all coordinates
            nuc_field = np.zeros([len(coordinates), 3], dtype=np.float64)
            nuc_coords = None
            nuc_charge = None
            for nucleus in quantum_subsystem.nuclei:
                if nucleus.index == nuc_index:
                    nuc_coords = nucleus.coordinate
                    nuc_charge = nucleus.charge
            for i, coord in enumerate(coordinates):
                r_ab = coord - nuc_coords
                nuc_field[i, 0] += tensor_tools.compute_interaction_tensor_element(distance_vector=r_ab,
                                                                                   multi_index=multi_index_x,
                                                                                   tensor_coefficients=constants.
                                                                                   values.tensor_coefficients)
                nuc_field[i, 1] += tensor_tools.compute_interaction_tensor_element(distance_vector=r_ab,
                                                                                   multi_index=multi_index_y,
                                                                                   tensor_coefficients=constants.
                                                                                   values.tensor_coefficients)
                nuc_field[i, 2] += tensor_tools.compute_interaction_tensor_element(distance_vector=r_ab,
                                                                                   multi_index=multi_index_z,
                                                                                   tensor_coefficients=constants.
                                                                                   values.tensor_coefficients)
            unique_f_cache[key].vals[comp] += nuc_field * nuc_charge
            contr_added = True
        # Set values_are_set to True if any contribution was added
        if contr_added:
            unique_f_cache[key].values_are_set = True


def calc_pert_ind_contr(ids: list,
                        comp: tuple,
                        unique_f_cache: dict,
                        fbf_p_tuples: list,
                        quantum_subsystem: subsystem.QuantumSubsystem,
                        classical_subsystem: subsystem.ClassicalSubsystem,
                        threshold: Optional[float],
                        max_iterations: Optional[int],
                        solver: Optional[str]) -> float:
    # Check if tuple is empty, i.e., unperturbed Field
    if not ids[0] or not ids[1]:
        # One of FBF is unperturbed, use already existing induced dipoles.
        if len(ids[0]) == 0:
            field = unique_f_cache[hash(ids[0])].vals[comp]
        else:
            field = unique_f_cache[hash(ids[1])].vals[comp]
        return np.einsum('ij, ij', field, classical_subsystem.induced_dipoles.induced_dipoles)
    # Check if one of the values are not set
    if not unique_f_cache[hash(ids[0])].values_are_set or not unique_f_cache[hash(ids[1])].values_are_set:
        # one of the unique F values are not set, i.e., the Field is 0 and the entire FBF term becomes 0.
        return 0.0

    # Its not unperturbed, and not zero, calculate component number for each F
    num_comps_f1 = 1
    for pert in fbf_p_tuples[0].p:
        if pert.o == 'EL':
            num_comps_f1 *= 3
        if pert.o == 'GEO':
            num_comps_f1 *= 3 * quantum_subsystem.num_nuclei
    num_comps_f2 = 1
    for pert in fbf_p_tuples[1].p:
        if pert.o == 'EL':
            num_comps_f2 *= 3
        if pert.o == 'GEO':
            num_comps_f2 *= 3 * quantum_subsystem.num_nuclei
    # Check which FB pair is cheaper
    if num_comps_f1 > num_comps_f2:
        # Use second term, i.e., BF2
        ind_dip = classical_subsystem.solve_induced_dipoles(threshold=threshold,
                                                            max_iterations=max_iterations,
                                                            solver=solver,
                                                            external_fields=unique_f_cache[hash(ids[1])].vals[comp],
                                                            perturbed=True)
        return np.einsum('ij, ij', unique_f_cache[hash(ids[0])].vals[comp], ind_dip)
    elif num_comps_f1 == num_comps_f2:
        # Is equal, therefore just use F1
        ind_dip = classical_subsystem.solve_induced_dipoles(threshold=threshold,
                                                            max_iterations=max_iterations,
                                                            solver=solver,
                                                            external_fields=unique_f_cache[hash(ids[0])].vals[comp],
                                                            perturbed=True)
        return np.einsum('ij, ij', unique_f_cache[hash(ids[1])].vals[comp], ind_dip)
    else:
        # Use first term, i.e., F1B
        # Is equal, therefore just use F1
        ind_dip = classical_subsystem.solve_induced_dipoles(threshold=threshold,
                                                            max_iterations=max_iterations,
                                                            solver=solver,
                                                            external_fields=unique_f_cache[hash(ids[0])].vals[comp],
                                                            perturbed=True)
        return np.einsum('ij, ij', unique_f_cache[hash(ids[1])].vals[comp], ind_dip)


def subsets_of_list(filled_list) -> list:
    """Identifies all the distinct ways the elements in a list can be grouped into non-empty subsets,
     where each element is included exactly once. The number of subsets generated is given through 2^(n-1), where n
     corresponds to the number of elements of the given list.

    Args:
        filled_list: List filled me elements (non-empty), for which subsets are determined.

    Returns:
        List of subsets (list of lists).
    """
    def backtrack(start, path):
        if start == len(filled_list):
            partitions.append(path[:])
            return
        for i in range(start, len(filled_list)):
            backtrack(i + 1, path + [filled_list[start:i + 1]])

    partitions = []
    backtrack(0, [])
    return partitions
