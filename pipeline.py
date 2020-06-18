#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

import sys
import re
import subprocess
import numpy as np
import time
from enum import Enum
from Condition import Condition

class Solver(Enum):
    PLINGELING = 0
    GLUCOSE_SYRUP = 1

def parse_input(path):
    with open(path, "r") as f:
        lines = np.asarray(f.readlines())
        start_indices = np.asarray([i for i, line in enumerate(lines) if "//" in line] + [len(lines) + 1])
        m = int(lines[start_indices[0] + 1][len("segsites: "):])
        n = start_indices[1] - start_indices[0] - 4
        mats = list()

        for i, j in enumerate(start_indices[:-1] + 3):
            mat = np.asarray([list(line.rstrip()) for line in lines[j:(start_indices[i+1] - 1)]], dtype='float64')
            mat = mat[:, ~np.all(mat == 0, axis = 0)]
            mats.append(mat)

    return mats

def gen_i_conditions(n, m, total_nodes, debug):
    num_internal_nodes = total_nodes - n - 1
    # Each leaf must be included in the tree (I(l, k, l) = True)
    root_i_condition = Condition([[1]], True, n*m, total_nodes-(n-1))

    i_condition = Condition([list(range(2, num_internal_nodes+2))], True, n*m, num_internal_nodes + 2)
    leaf_i_condition = Condition([[num_internal_nodes+2]], True, n*m, num_internal_nodes + 2)

    final_i_val = n * m * (num_internal_nodes + 2)

    if debug:
        print("i vars start at: 1")

    return [root_i_condition, i_condition, leaf_i_condition], final_i_val

def gen_t_conditions(n, m, num_internal_nodes, total_nodes, final_i_var, debug):
    # m commodities and n+m internal nodes
    # commodities can't be repeated over using the Condition (requires adding a different number to first elements than last), but each node can be
    root_t_condition = Condition([[-1*(final_i_var + 1)]], True, m, total_nodes)
    t_condition = Condition(list(), False)
    leaf_t_condition = Condition([[x] for x in range(final_i_var + num_internal_nodes + 2, final_i_var + total_nodes + 1)], True, m, total_nodes)

    for k in range(m):
        for j in range(num_internal_nodes):
            i_vars = list()
            t_var = final_i_var + 2 + k*total_nodes + j
            for l in range(n):
                i_var = 2 + (n*k+l)*(total_nodes - n + 1) + j
                t_condition.add_clause([-i_var, t_var])
                i_vars.append(i_var)
            t_condition.add_clause([x for x in i_vars] + [-t_var])

    final_t_var = final_i_var + m*total_nodes

    if debug:
        print("t vars start at:", final_i_var + 1)

    return [root_t_condition, t_condition, leaf_t_condition], final_t_var

def gen_f_conditions(n, m, num_internal_nodes, total_edges, final_t_var, debug):
    f_condition_1 = Condition([list(range(final_t_var+1, final_t_var+num_internal_nodes+1))], True, m*n, total_edges - (num_internal_nodes)*(n-1))
    f_condition_2 = Condition(list(), False)

    f_vars = list(range(final_t_var + 1, final_t_var + total_edges - (num_internal_nodes)*(n-1) + 1))

    if debug:
        print("f vars start at:", f_vars[0])

    for k in range(m):
        for l in range(n):
            for j in range(1, num_internal_nodes+1):
                current_node_f_vars = [final_t_var + (k*n+l)*(total_edges-(num_internal_nodes)*(n-1)) + j]
                current_i_var = (num_internal_nodes + 2)*(l+k*n) + j + 1
                start_i_var = current_i_var - j
                f_condition_2.add_clause([-1*current_node_f_vars[-1], current_i_var])

                for i in range(j-1):
                    current_node_f_vars.append(current_node_f_vars[-1] + num_internal_nodes - max(i,1))
                    f_condition_2.add_clause([-1*current_node_f_vars[-1], current_i_var])

                f_condition_2.add_clause(current_node_f_vars + [-1*current_i_var])

            # Condition saying that there must be flow from at least one internal node to the leaf we're currently concerned with
            # The condition for *only* one edge going to the leaf comes in F condition 3
            f_condition_2.add_clause([x+1 for x in current_node_f_vars[1:]] + [current_node_f_vars[-1] + 2])

    f_condition_3 = Condition(list(), True, m*n, total_edges-(num_internal_nodes)*(n-1))

    for j in range(2, num_internal_nodes+2):
        start_f_var = final_t_var + j

        for start_i in range(j - 1):
            next_f_var = start_f_var + num_internal_nodes - max(start_i, 1)

            if start_i == 0 and j == m+n+1:
                # no edge from the root to any leaf, so continue
                start_f_var = start_f_var + num_internal_nodes - max(start_i, 1)
                continue

            for i in range(start_i + 1, j):
                f_condition_3.add_clause([-1*start_f_var, -1*next_f_var])
                next_f_var = next_f_var + num_internal_nodes - max(i, 1)

            start_f_var = start_f_var + num_internal_nodes - max(start_i, 1)
    
    f_condition_4 = Condition(list(), True, m*n, total_edges-(num_internal_nodes)*(n-1))
    current_f_vars = f_vars

    for i in range(m+n): # don't need to look at edges out of last internal node bc there's only one, going to leaf l
        for start_j in range(min(num_internal_nodes, num_internal_nodes-i+1)):
            first_f_var = current_f_vars[start_j]

            for j in range(start_j + 1, min(num_internal_nodes, num_internal_nodes-i+1)):
                second_f_var = current_f_vars[j]
                f_condition_4.add_clause([-1*first_f_var, -1*second_f_var])
        
        current_f_vars = current_f_vars[min(num_internal_nodes, num_internal_nodes-i+1):]


    f_condition_5 = Condition(list(), True, n*m, total_edges-(num_internal_nodes)*(n-1))
    current_f_vars = f_vars[num_internal_nodes:]


    for i_prime in range(num_internal_nodes):
        start_f_var = f_vars[i_prime]
        current_f_i_prime_vars = f_vars[num_internal_nodes:]
        f_i_prime_vars = [f_vars[i_prime]]


        for i in range(i_prime):
            f_i_prime_vars.append(current_f_i_prime_vars[i_prime-i-1])
            current_f_i_prime_vars = current_f_i_prime_vars[min(num_internal_nodes, num_internal_nodes-i):]

        for f_i in current_f_vars[:min(num_internal_nodes, num_internal_nodes-i_prime)]: # second arg to min is simplified from m+n+2-(i+1)               
            f_condition_5.add_clause([-1*(f_i)] + [x for x in f_i_prime_vars])

        current_f_vars = current_f_vars[min(num_internal_nodes, num_internal_nodes-i_prime):]



    last_f_var = final_t_var + m*n*(f_vars[-1] - final_t_var)

    return [f_condition_1, f_condition_2, f_condition_3, f_condition_4, f_condition_5], last_f_var, f_vars

def gen_x_conditions(n, m, num_internal_nodes, total_edges, last_f_var, f_vars, debug):
    x_condition = Condition(list(), False)
    x_vars = [x for x in range(last_f_var + 1, last_f_var + total_edges + 1)]
    leaf_f_vars = [0 for x in range(num_internal_nodes)]
    last_index = len(f_vars) - 1

    if debug:
        print("x vars start at:", x_vars[0])

    for f in range(num_internal_nodes):
        leaf_f_vars[num_internal_nodes-f-1] = f_vars[-1-f*(f+1)//2]

    last_x_var = last_f_var + m*total_edges

    for k in range(m):
        x_i = 0

        for i, f_var in enumerate(f_vars):
            if f_var in leaf_f_vars:
                for x in range(n):
                    x_condition.add_clause([(n*k+x)*(total_edges-(num_internal_nodes)*(n-1)) + f_var, -1*(x_vars[x_i + i +x] + k*total_edges)])
                    x_condition.add_clause([-1*((n*k+x)*(total_edges-(num_internal_nodes)*(n-1)) + f_var), x_vars[x_i + i +x] + k*total_edges])

                x_i += n-1
            else:
                for l in range(n):
                    x_condition.add_clause([-1*((n*k+l)*(total_edges-(num_internal_nodes)*(n-1)) + f_var), x_vars[x_i+i]+k*total_edges])

                x_condition.add_clause([(n*k+x)*(total_edges-(num_internal_nodes)*(n-1)) + f_var for x in range(n)] + [-1*(x_vars[x_i+i] + k*total_edges)])

    #last_x_var = x_vars[-1]

    return [x_condition], last_x_var, x_vars

def gen_d_conditions(n, m, num_internal_nodes, total_edges, last_x_var, x_vars, debug):
    d_condition_1 = Condition(list(), False)
    d_condition_2 = Condition(list(), False)
    d_vars = list(range(last_x_var + 1, last_x_var + total_edges + 1))

    if debug:
        print("d vars start at:", d_vars[0])

    for i, x_var in enumerate(x_vars):
        for k in range(m):
            d_condition_1.add_clause([-1*(k*total_edges + x_var), d_vars[i]])
        d_condition_1.add_clause([x_vars[i] + x*total_edges for x in range(m)] + [-1*d_vars[i]])
    
    for j in range(3, num_internal_nodes + n + 1):
        d_i_vars = d_vars

        for i in range(min(num_internal_nodes-1,j-2)):
            if i == 0 and j > num_internal_nodes:
                d_i_vars = d_i_vars[num_internal_nodes:]
                continue

            d_ij = d_i_vars[j - i - 1]

            if i == 0:
                d_h_vars = d_i_vars[num_internal_nodes:]
            else:
                d_h_vars = d_i_vars[n + num_internal_nodes - i:]

            for h in range(i+1, min(num_internal_nodes,j-1)):

                d_hj = d_h_vars[j - h - 1]
                d_g_vars = d_h_vars[n + num_internal_nodes - h:]                

                for g in range(h+1, min(num_internal_nodes+1, j)):
                    
                    d_gj = d_g_vars[j - g - 1]
                    d_condition_2.add_clause([-d_ij, -d_hj, -d_gj])
                    d_g_vars = d_g_vars[n + num_internal_nodes - g:]
                
                d_h_vars = d_h_vars[n + num_internal_nodes - h:]
            
            if i == 0:
                d_i_vars = d_i_vars[num_internal_nodes:]
            else:
                d_i_vars = d_i_vars[n + num_internal_nodes - i:]

    return [d_condition_1, d_condition_2], d_vars[-1]

def gen_tree_conditions(n, m, num_internal_nodes, debug):
    # Have adjacency mat for edges?
    # characters = columns
    # taxa = rows
    total_nodes = 1 + num_internal_nodes + n # root + internal + leaves
    total_edges = num_internal_nodes + (num_internal_nodes*(num_internal_nodes-1))//2 + num_internal_nodes*n

    i_conditions, final_i_var = gen_i_conditions(n, m, total_nodes, debug) # Conditions for each node being included in a commodity tree with flow passing through them to a specific leaf
    t_conditions, final_t_var = gen_t_conditions(n, m, num_internal_nodes, total_nodes, final_i_var, debug) # Conditions for each node being included in a commodity tree
    f_conditions, final_f_var, f_vars = gen_f_conditions(n, m, num_internal_nodes, total_edges, final_t_var, debug) # Conditions for each edge being included in a commodity tree going to a specific leaf
    x_conditions, final_x_var, x_vars = gen_x_conditions(n, m, num_internal_nodes, total_edges, final_f_var, f_vars, debug) # Condition for edges being included in a commodity tree
    d_conditions, final_d_var = gen_d_conditions(n, m, num_internal_nodes, total_edges, final_x_var, x_vars, debug) # Condition for edges being included in the DAG

    conditions = i_conditions + t_conditions + f_conditions + x_conditions + d_conditions
    return conditions, final_t_var, final_d_var


def gen_rct_conditions(num_rows, num_cols, num_internal_nodes, total_nodes, final_edge_var, debug):
    final_rct_var = final_edge_var + num_cols*total_nodes
    rct_vars = [r for r in range(final_edge_var + 1, final_rct_var + 1)]
    i_offset = num_rows*num_cols*(num_internal_nodes + 2)

    if debug:
        print("rct vars start at:", rct_vars[0])

    rct_condition_1 = Condition([[i_offset + 1, -1*(final_edge_var + 1)]], True, num_cols*total_nodes, 1)
    rct_condition_2 = Condition([[x for x in range(final_edge_var + 2, final_edge_var + num_internal_nodes+2)]], True, num_cols, total_nodes)
    
    # Root can't be the root of the subtree
    rct_condition_2.add_clause([-1*(final_edge_var + 1)])

    for x in range(num_internal_nodes + 2, total_nodes + 1):
        rct_condition_2.add_clause([-1*(x+final_edge_var)])

    rct_condition_3 = Condition(list(), True, num_cols, total_nodes)

    for i in range(2, total_nodes):
        for j in range(i+1, total_nodes + 1):
            rct_condition_3.add_clause([-1*(final_edge_var + i), -1*(final_edge_var + j)])

    return [rct_condition_1, rct_condition_2, rct_condition_3], rct_vars


def gen_z_conditions(num_rows, num_cols, num_edges, num_internal_nodes, total_nodes, final_edge_var, final_rct_var, debug):
    final_z_var = final_rct_var + num_cols * num_edges
    final_ct_var = final_z_var + num_cols * total_nodes
    z_vars = [z for z in range(final_rct_var + 1, final_z_var + 1)]
    x_vars = [x for x in range(final_edge_var - (num_cols+1)*num_edges + 1, final_edge_var - num_edges + 1)]
    ct_vars = [c for c in range(final_z_var + 1, final_ct_var + 1)]

    if debug:
        print("z vars start at: {}".format(z_vars[0]))

    z_condition = Condition(list(), False)
    clauses = list()

    for k in range(num_cols):
        offset = k*num_edges
        for i in range(num_internal_nodes):
            for j in range(i + 1, total_nodes):
                z_var = z_vars[offset]
                ct_var = ct_vars[i + k*total_nodes]
                x_var = x_vars[offset]

                if i == 0 and j > (num_internal_nodes):
                    continue
                elif i == 0:
                    z_condition.add_clause([-z_var])
                    offset += 1
                    continue

                z_condition.add_clause([ct_var, -z_var])
                z_condition.add_clause([x_var, -z_var])
                z_condition.add_clause([-ct_var, -x_var, z_var])

                offset += 1


    return [z_condition], z_vars


def gen_ct_conditions(input, num_rows, num_cols, num_edges, num_internal_nodes, total_nodes, rct_vars, z_vars, debug):
    final_z_var = z_vars[-1]
    final_ct_var = final_z_var + num_cols * total_nodes
    ct_vars = [c for c in range(final_z_var + 1, final_ct_var + 1)]
    ct_condition = Condition(list(), False)

    if debug:
        print("ct vars start at: {}".format(ct_vars[0]))

    for k in range(num_cols):
        ct_condition.add_clause([-ct_vars[k*total_nodes]])

        for j in range(1,total_nodes):
            ct_var  = ct_vars[j + k*total_nodes]
            rct_var = rct_vars[j+k*total_nodes]
            clause = [-ct_var, rct_var]
            ct_condition.add_clause([ct_var, -rct_var])

            if j < num_internal_nodes+1:
                start = 0
                stop = j
                z_offset = k*num_edges + j - 1
            else:
                start = 1
                stop = num_internal_nodes+1
                z_offset = k*num_edges + num_internal_nodes + j - 2

            for i in range(start, stop):
                z_var = z_vars[z_offset]
                clause.append(z_var)
                ct_condition.add_clause([ct_var, -z_var])

                if i == 0:
                    z_offset += num_internal_nodes - 1
                else:
                    z_offset += num_internal_nodes + num_rows - i - 1
            
            ct_condition.add_clause(clause)
    
    leaf_ct_condition = Condition(list(), False)

    last_ct_internal_var = z_vars[-1] + num_internal_nodes+1
    non_zero_indices = input.nonzero()

    for k in range(num_cols):
        true_ct_vars = non_zero_indices[0][np.where(non_zero_indices[1] == k)]
        
        for l in range(num_rows):
            if l in true_ct_vars:
                leaf_ct_condition.add_clause([last_ct_internal_var + k*total_nodes + l + 1])
            else:
                leaf_ct_condition.add_clause([-1*(last_ct_internal_var + k*total_nodes + l + 1)])
    
    return [ct_condition, leaf_ct_condition], final_ct_var

def gen_subtree_conditions(input, n, m, num_edges, num_internal_nodes, final_node_var, final_edge_var, debug):
    conditions = list()
    total_nodes = num_internal_nodes + n + 1
    rct_conditions, rct_vars = gen_rct_conditions(n, m, num_internal_nodes, total_nodes, final_edge_var, debug)
    z_conditions, z_vars = gen_z_conditions(n, m, num_edges, num_internal_nodes, total_nodes, final_edge_var, rct_vars[-1], debug)
    ct_conditions, final_ct_var = gen_ct_conditions(input, n, m, num_edges, num_internal_nodes, total_nodes, rct_vars, z_vars, debug)

    return rct_conditions + z_conditions + ct_conditions, final_ct_var

def gen_reticulation_conditions(n, m, num_internal_nodes, num_edges, final_d_var, final_ct_var, debug):
    r_vars = [x for x in range(final_ct_var +  1, final_ct_var + num_internal_nodes + n + 1)] # r variables for all internal and leaf nodes but not the root bc it's a source
    leaf_r_vars = r_vars[-n:]
    d_vars = [x for x in range(final_d_var - num_edges + 1, final_d_var + 1)]
    final_r_var = r_vars[-1]

    if debug:
        print("r vars start at: {}".format(r_vars[0]))

    r_condition = Condition(list(), False)

    r_condition.add_clause([-1*(final_ct_var + 1)])

    j_node = 2
    offset = 1

    for r in r_vars[1:]:
        clauses = list()
        vars = list()

        for i in range(min(j_node, num_internal_nodes+1)):
            if i == 0 and j_node > num_internal_nodes:
                offset += num_internal_nodes-1
                continue # no edge from 0 to leaves

            vars.append(d_vars[offset])

            if i == 0:
                offset += num_internal_nodes-1
            else:
                offset +=  n + num_internal_nodes - (i + 1)

        r_implies_dnf = list()
        clause = list()
        dnf_implies_r = list()

        for x in range(len(vars)):
            clause = list()
            for y in range(len(vars)):
                if y == x:
                    continue
                clause.append(vars[y])
            r_implies_dnf.append(clause +  [-r])

        for x, var in enumerate(vars[:-1]):
            for y in range(x + 1, len(vars)):
                dnf_implies_r.append([-var, -vars[y], r])
        
        clauses = r_implies_dnf + dnf_implies_r
        j_node += 1
        offset = j_node-1


        for clause in clauses:
            r_condition.add_clause(clause)

    
    for leaf in leaf_r_vars:
        r_condition.add_clause([-leaf])

    return [r_condition], final_r_var


def gen_counting_conditions(n, m, num_internal_nodes, goal_count, final_r_var, debug):
    r_vars = [r for r in range(final_r_var - (num_internal_nodes+n - 1), final_r_var + 1)]
    c_vars = [c for c in range(final_r_var + 1, final_r_var + (len(r_vars) + 1)*(goal_count + 1) + 1)]
    c_condition = Condition(list(), False)
    num_r_vars = len(r_vars)
    num_c_vars = len(c_vars)

    for k in range(goal_count + 1):
        for i in range(num_r_vars):
            current_c_var_index = k*(num_r_vars + 1) + i
            c_condition.add_clause([-c_vars[current_c_var_index], c_vars[current_c_var_index + 1]])

            if k < goal_count:
                c_condition.add_clause([-c_vars[current_c_var_index], -r_vars[i], c_vars[current_c_var_index + num_r_vars + 2]])
                # sympy_clauses = sympy_clauses & Implies(c_vars[current_c_var_index] & r_vars[i], c_vars[(k+1)*(m+n+1) + i + 1])
            if k == 0:
                c_condition.add_clause([-r_vars[i], c_vars[current_c_var_index + 1]])
                # sympy_clauses = sympy_clauses & Implies(r_vars[i], c_vars[current_c_var_index + 1])

    final_c_var = c_vars[-1]

    c_condition.add_clause([-1*final_c_var])

    if debug:
        print("c vars start at: {}".format(c_vars[0]))
        print("c vars end at: {}".format(c_vars[-1]))

    return [c_condition], final_c_var


def write_cnf_file(conditions, num_vars, num_clauses, cnf_file_path):
    with open(cnf_file_path, "w+") as f:
        print("p cnf", num_vars, num_clauses, file=f)

        for condition in conditions:
            condition.write_condition(f)

def append_to_cnf_file(conditions, num_vars, num_clauses, cnf_file_path):
    with open(cnf_file_path, "w+") as f:
        s = "p cnf " + str(num_vars) + " " + str(num_clauses) + "\n"
        f.write(s)

        with open("temp") as temp:
            old_conditions = temp.readlines()
            f.writelines(old_conditions[1:]) # don't write the old header
        
        for condition in conditions:
            condition.write_condition(f)

def get_num_clauses(conditions):
    num_clauses = 0

    for condition in conditions:
        num_clauses += len(condition.clauses) * condition.num_repeats

    return num_clauses

def call_solver(solver_path, cnf_file_path, time_limit):

    start_time = time.time()
    #result = subprocess.run([solver_path, "-nthreads=12", cnf_file_path], capture_output=True)
    try:
        result = subprocess.run([solver_path, "-nthreads=12", cnf_file_path], timeout=time_limit, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as e:
        print("SAT solver timed out after {} seconds".format(time_limit))
        return time_limit, False

    end_time = time.time()

    total_time = end_time - start_time
    sat = " SATISFIABLE" in str(result.stdout)

    return total_time, sat

def gen_conditions(num_rows, num_cols, mat, num_internal_nodes, num_edges, bound, debug):
    tree_conditions, final_node_var, final_edge_var = gen_tree_conditions(num_rows, num_cols, num_internal_nodes, debug)
    subtree_conditions, final_ct_var = gen_subtree_conditions(mat, num_rows, num_cols, num_edges, num_internal_nodes, final_node_var, final_edge_var, debug)
    reticulation_conditions, final_r_var = gen_reticulation_conditions(num_rows, num_cols, num_internal_nodes, num_edges, final_edge_var, final_ct_var, debug)
    counting_conditions, final_c_var = gen_counting_conditions(num_rows, num_cols, num_internal_nodes, bound, final_r_var, debug)
    conditions = tree_conditions + subtree_conditions + reticulation_conditions + counting_conditions
    
    return conditions, final_c_var

def minimize_sat(num_rows, num_cols, num_internal_nodes, initial_bound, input_mat, solver, cnf_file_path, time_limit, debug):
    total_time = 0
    runs_required = 0
    sat = True
    bound = initial_bound

    if (solver == Solver.GLUCOSE_SYRUP):
        solver_path = "./glucose-syrup/parallel/glucose-syrup"
    else:
        solver_path = "./lingeling/plingeling"

    while sat and bound >= 0 and bound <= initial_bound:
        num_edges = (num_internal_nodes) + num_rows * (num_internal_nodes) + (num_internal_nodes)*(num_internal_nodes-1)//2
        conditions, final_var = gen_conditions(num_rows, num_cols, input_mat, num_internal_nodes, num_edges, bound, debug)
        num_clauses = get_num_clauses(conditions)
        write_cnf_file(conditions, final_var, num_clauses, cnf_file_path)
        time, sat = call_solver(solver_path, cnf_file_path, time_limit)
        runs_required += 1
        total_time += time
        print("bound {}, {}, time so far: {}".format(bound, "SAT" if sat else "UNSAT", total_time))

        if sat:
            bound -= 1
        else:
            bound += 1

        if num_internal_nodes > num_rows - 2 + 2*bound:
            num_internal_nodes = max(num_rows - 2 + 2*bound, 2)

    if bound == -1:
        bound = 0
    if bound > initial_bound:
        bound = None

    results = {"time":total_time, "bound":bound, "runs_required":runs_required, "method":"SAT"}
    # subprocess.run(["rm", "temp"])
    return results

def print_results(results, input_name, outdir):
    if outdir != None:
        with open(outdir + "/" + input_name, "w+") as f: 
            print("\n----results for {}-----".format(input_name), file=f)

            for result in results:
                print("\nresults for {}:".format(result["method"]), file=f)
                print("  time taken: {}".format(result["time"]), file=f)
                print("  bound: {}".format(result["bound"]), file=f)

                if 'runs_required' in result.keys():
                    print("  runs required: {}".format(result['runs_required']), file=f)

            print("\n", file=f)

    print("\n----results for {}-----".format(input_name))
    for result in results:
        print("\nresults for {}:".format(result["method"]))
        print("  time taken: {}".format(result["time"]))
        print("  bound: {}".format(result["bound"]))

        if 'runs_required' in result.keys():
            print("  runs required: {}".format(result['runs_required']))

    print("\n")

def main(argv):
    outdir = "./output"
    solver = Solver.GLUCOSE_SYRUP
    time_limit = 3600
    debug = False

    if len(argv) < 2 or "-h" in argv or "--help" in argv:
        print("\nError: Usage\n\tpython3 pipeline.py [input file] <options>")
        print("\nOptions: ")
        print("\t-h/--help\tDisplay this documentation.\n")
        print("\t-d\tPrint debug messages for where each variable starts.\n")
        print("\t-o\tDirectory the output should go to. If not specified, \n\t\toutput is printed to stdout.\n")
        print("\t-t\tTime limit in seconds for each SAT solver run. \n\t\tDefault is 3600.\n")
        print("\t-n\tNumber of internal nodes to build the initial graph with. \n\t\tDefault is n + m where n = number of rows, \n\t\tm = number of cols.\n")
        print("\t-b\tInitial upper bound. Default is n + m where \n\t\tn = number of rows, m = number of cols.\n")

        return
        
    input_file = argv[1]

    if "-o" in argv:
        outdir = argv[argv.index("-o") + 1]
    if "-t" in argv:
        time_limit = int(argv[argv.index("-t") + 1])
    if "-d" in argv:
        debug = True

    input_path = "./input/" + input_file
    input_matrices = parse_input(input_path)
    i = 0

    for mat in input_matrices:
        n = mat.shape[0]
        m = mat.shape[1]

        if "-n" in argv:
            num_internal_nodes = int(argv[argv.index("-n") + 1])
        elif "-b" in argv:
            num_internal_nodes = n - 2 + 2*int(argv[argv.index("-b") + 1])
        else:
            num_internal_nodes = n + m
        if "-b" in argv:
            bound = min(int(argv[argv.index("-b") + 1]), num_internal_nodes)
        else:
            bound = min(n + m, num_internal_nodes)

        input_name = input_file + "_" + str(i)

        sat_results = minimize_sat(n, m, num_internal_nodes, bound, mat, solver, input_name + ".cnf", time_limit, debug)
        with open("./input/temp", "w+") as temp:
            s = ""
            for row in mat:
                for entry in row:
                    s += str(int(entry))
                s += "\n"
            temp.write(s)

        print_results([sat_results], input_name, outdir)

        i += 1 
    
    return

# main(["pipeline.py", "test5x5", "-d"])
main(sys.argv)
