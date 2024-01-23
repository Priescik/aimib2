import copy
import sys
from matplotlib import tri

import tqdm
sys.path.insert(0,'..')

import argparse
import os
import sys
import time
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib
import frams
from helpers import eaSimple_modified
import matplotlib.pyplot as plt
import pickle
from framsfiles import writer as framswriter
from deap.algorithms import varAnd
import matplotlib.pyplot as plt


# Note: this may be less efficient than running the evolution directly in Framsticks, so if performance is key, compare both options.


def genotype_within_constraint(genotype, dict_criteria_values, criterion_name, constraint_value):
    REPORT_CONSTRAINT_VIOLATIONS = False
    if constraint_value is not None:
        actual_value = dict_criteria_values[criterion_name]
        if actual_value > constraint_value:
            if REPORT_CONSTRAINT_VIOLATIONS:
                print('Genotype "%s" assigned low fitness because it violates constraint "%s": %s exceeds threshold %s' % (
                    genotype, criterion_name, actual_value, constraint_value))
            return False
    return True


def frams_evaluate(frams_cli, individual):
    # fitness of -1 is intended to discourage further propagation of this genotype via selection ("this genotype is very poor")
    BAD_FITNESS = [-1] * len(OPTIMIZATION_CRITERIA)
    # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    genotype = individual[0]
    data = frams_cli.evaluate([genotype])
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    valid = True
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit]
                   for crit in OPTIMIZATION_CRITERIA]
    # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
    except (KeyError, TypeError) as e:
        valid = False
        # print('Problem "%s" so could not evaluate genotype "%s", hence assigned it low fitness: %s' % (str(e), genotype, BAD_FITNESS))
        # print('Problem "%s" so could not evaluate genotype in format %s' % (str(e), genotype[:3]))
    if valid:
        default_evaluation_data['numgenocharacters'] = len(
            genotype)  # for consistent constraint checking below
        valid &= genotype_within_constraint(
            genotype, default_evaluation_data, 'numparts', parsed_args.max_numparts)
        valid &= genotype_within_constraint(
            genotype, default_evaluation_data, 'numjoints', parsed_args.max_numjoints)
        valid &= genotype_within_constraint(
            genotype, default_evaluation_data, 'numneurons', parsed_args.max_numneurons)
        valid &= genotype_within_constraint(
            genotype, default_evaluation_data, 'numconnections', parsed_args.max_numconnections)
        valid &= genotype_within_constraint(
            genotype, default_evaluation_data, 'numgenocharacters', parsed_args.max_numgenochars)
    if not valid:
        fitness = BAD_FITNESS
    return fitness


def frams_crossover(frams_cli, individual1, individual2):
    # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    geno1 = individual1[0]
    # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    geno2 = individual2[0]
    individual1[0] = frams_cli.crossOver(geno1, geno2)
    individual2[0] = frams_cli.crossOver(geno1, geno2)
    return individual1, individual2


def frams_mutate(frams_cli, individual):
    # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    individual[0] = frams_cli.mutate([individual[0]])[0]
    return individual,


def frams_getsimplest(frams_cli, genetic_format, initial_genotype):
    return initial_genotype if initial_genotype is not None else frams_cli.getSimplest(genetic_format)


def prepareToolbox(frams_cli, tournament_size, genetic_format, initial_genotype):
    creator.create("FitnessMax", base.Fitness, weights=[
                   1.0] * len(OPTIMIZATION_CRITERIA))
    # would be nice to have "str" instead of unnecessary "list of str"
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_cli,
                     genetic_format, initial_genotype)  # "Attribute generator"
    # (failed) struggle to have an individual which is a simple str, not a list of str
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
    # https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
    # https://github.com/DEAP/deap/issues/339
    # https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
    # https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", frams_evaluate, frams_cli)
    toolbox.register("mate", frams_crossover, frams_cli)
    toolbox.register("mutate", frams_mutate, frams_cli)
    if len(OPTIMIZATION_CRITERIA) <= 1:
        toolbox.register("select", tools.selTournament,
                         tournsize=tournament_size)
    else:
        toolbox.register("select", tools.selNSGA2)
    toolbox.register("save_hof", save_tool)
    return toolbox


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
    parser.add_argument('-path', type=ensureDir, required=True,
                        help='Path to Framsticks CLI without trailing slash.')
    parser.add_argument('-lib', required=False,
                        help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim",
                        help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

    parser.add_argument('-genformat', required=False,
                        help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.')
    parser.add_argument('-initialgenotype', required=False,
                        help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.')

    parser.add_argument('-opt', required=True, help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria optimization, separate the names by the comma.')
    parser.add_argument('-popsize', type=int, default=50,
                        help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5,
                        help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5,
                        help="Tournament size, default: 5.")
    parser.add_argument('-pmut', type=float, default=0.9,
                        help="Probability of mutation, default: 0.9")
    parser.add_argument('-pxov', type=float, default=0.2,
                        help="Probability of crossover, default: 0.2")
    parser.add_argument('-hof_size', type=int, default=10,
                        help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_savefile', required=False,
                        help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen).')

    parser.add_argument('-max_numparts', type=int, default=None,
                        help="Maximum number of Parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None,
                        help="Maximum number of Joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None,
                        help="Maximum number of Neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None,
                        help="Maximum number of Neural connections. Default: no limit")
    parser.add_argument('-max_numgenochars', type=int, default=None,
                        help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit")
    
    parser.add_argument('-series', default='x')

    return parser.parse_args()


def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def save_tool(hof, clear):
    mode = "w" if clear else "a"
    with open(f'HoFs/HoF-f{parsed_args.series}', mode) as outfile:
        for ind in hof:
            keyval = {}
            # construct a dictionary with criteria names and their values
            for i, k in enumerate(OPTIMIZATION_CRITERIA):
                # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
                keyval[k] = ind.fitness.values[i]
            # Note: prior to the release of Framsticks 5.0, saving e.g. numparts (i.e. P) without J,N,C breaks re-calcucation of P,J,N,C in Framsticks and they appear to be zero (nothing serious).
            outfile.writelines(framswriter.from_collection(
                {"_classname": "org", "genotype": ind[0], **keyval}))
            outfile.writelines("\n")


def save_genotypes(filename, OPTIMIZATION_CRITERIA, hof, timer):
    with open(filename, "w") as outfile:
        for ind in hof:
            keyval = {}
            # construct a dictionary with criteria names and their values
            for i, k in enumerate(OPTIMIZATION_CRITERIA):
                # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
                keyval[k] = ind.fitness.values[i]
            # Note: prior to the release of Framsticks 5.0, saving e.g. numparts (i.e. P) without J,N,C breaks re-calcucation of P,J,N,C in Framsticks and they appear to be zero (nothing serious).
            outfile.write(framswriter.from_collection(
                {"_classname": "org", "genotype": ind[0], **keyval}))
            outfile.write("\n")

        outfile.write(f'\n# {timer}')
    print("Saved '%s' (%d)" % (filename, len(hof)))


if __name__ == "__main__":
    # random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
    # must be set before FramsticksLib() constructor call
    FramsticksLib.DETERMINISTIC = False
    parsed_args = parseArguments()
    print("Argument values:", ", ".join(
        ['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    OPTIMIZATION_CRITERIA = parsed_args.opt.split(",")
    framsLib = FramsticksLib(
        parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("stddev", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    N_SERIES = 10
    # N_MUTANTS = 2
    MAX_N_PARENTS = 50 # also capped by hof size

    np.set_printoptions(precision=2)
    
    # parent_fits = np.zeros((4, N_SERIES * MAX_N_PARENTS))
    parent_fits = [[],[],[],[]]
    parents = [[],[],[],[]]

    debug = True

    fig, axs = plt.subplots(2,2)
    for format_str, format_n in zip(['0', '1', '4', 'H'], [0,1,2,3]):
        # org_id = -1
        for series in range(1,N_SERIES+1):
            file_path = f"HoFs/HoF-f{format_str}-{series}.gen"
            with open(file_path, 'r') as f:
                HoF = f.read().split('\n')
                in_genotype = False
                genotype = ""
                velocity = -1
                serie_limit_counter = 0
                for row in HoF:
                    if "org:" in row:
                        pass
                        # org_id += 1
                        # serie_limit_counter += 1
                        # if serie_limit_counter > MAX_N_PARENTS:
                        #     break
                    elif "genotype:" in row:
                        if format_str == '1' or format_str == '4':
                            genotype = row[9:]
                        else:
                            genotype = ""
                            in_genotype = True
                    elif "~" == row:
                        in_genotype = False
                    elif in_genotype:
                        genotype += row+"\n"
                    elif "velocity:" in row:
                        velocity = row[9:]
                        if float(velocity) <= 0: 
                            break # '0' and '-1' fitness individuals may be at the end of HoF file
                        parent_fits[format_n].append(float(velocity))
                        toolbox = prepareToolbox(framsLib, parsed_args.tournament, format_str, genotype)
                        parents[format_n].append(toolbox.individual())

        parents[format_n] = sorted(parents[format_n], 
                                   key=lambda x: parent_fits[format_n][parents[format_n].index(x)],
                                   reverse=True)
        parent_fits[format_n] = sorted(parent_fits[format_n], reverse=True)

        hist, brackets = None, None
        for bins in range(MAX_N_PARENTS, MAX_N_PARENTS*2, 1):
            hist, brackets = np.histogram(parent_fits[format_n], bins=bins)
            if (hist > 0).sum() >= MAX_N_PARENTS:
                break

        index = 0
        while len(parent_fits[format_n]) - 1 > index: #
            hist_idx = np.digitize(parent_fits[format_n][index], brackets, right=True) - 1
            if hist_idx == -1:
                hist_idx = 0
            if hist[hist_idx] > 1:
                del parent_fits[format_n][index]
                del parents[format_n][index]
                hist[hist_idx] -= 1
            else:
                index += 1

        print(len(parents[format_n]))

        children = [[None for _ in parents[format_n]] for _ in parents[format_n]]
        children_fits = [[0 for _ in parents[format_n]] for _ in parents[format_n]]

        for p_id, genotype in tqdm.tqdm(enumerate(parents[format_n])):
            for p_id2, genotype2 in enumerate(parents[format_n]):
                if p_id2 <= p_id: continue
                toolbox = prepareToolbox(framsLib, parsed_args.tournament, format_str, genotype)
                # print(parent_idx, parent2_idx)
                children[p_id][p_id2], children[p_id2][p_id] = toolbox.mate(copy.deepcopy(genotype), copy.deepcopy(genotype2))

                fit1 = toolbox.evaluate(children[p_id][p_id2])
                children[p_id][p_id2].fitness.values = fit1
                children_fits[p_id][p_id2] = fit1[0]

                fit2 = toolbox.evaluate(children[p_id2][p_id])
                children[p_id2][p_id].fitness.values = fit2
                children_fits[p_id2][p_id] = fit2[0]
                
        x_coords = []
        y_coords = []
        values = []
        inv_x = []
        inv_y = []
        changes = []
        changes2 = []
        for i in range(len(parents[format_n])):
            for j in range(i+1, len(parents[format_n])):
                f = 0
                denom = 0
                par_f = max(parent_fits[format_n][i], parent_fits[format_n][j])
                if children_fits[i][j] != -1: 
                    f += children_fits[i][j]
                    denom += 1
                if children_fits[j][i] != -1: 
                    f += children_fits[j][i]
                    denom += 1
                if denom == 0:
                    continue
                elif denom == 2:
                    f /= denom
                
                diff = f - par_f

                changes.append(f/par_f)
                changes2.append((f-par_f)/par_f)

                x_coords.append(parent_fits[format_n][i])
                y_coords.append(parent_fits[format_n][j])
                values.append(f)
                
                x_coords.append(parent_fits[format_n][j])
                y_coords.append(parent_fits[format_n][i])
                values.append(f)

        triang = tri.Triangulation(x_coords, y_coords)

        print('c1:', np.mean(changes), 'c2:', np.mean(changes2))

        im = axs[format_n%2, format_n//2].tripcolor(triang, values, cmap='jet')
        im.set_clim(0, max(parent_fits[format_n])*1.1)
        fig.colorbar(im, ax=axs[format_n%2, format_n//2])
        axs[format_n%2, format_n//2].set_title(f'format: {format_str}', loc='left')
        # axs[format_n%2, format_n//2].colorbar()
        # plt.show()
        # axs[format_n%2, format_n//2].scatter(x_coords, y_coords, c=values, cmap='viridis', s=2)
    print("job done")
    plt.show()