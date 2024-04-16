import pickle
import os
from os import path
import sys
from hashlib import sha256
import logging
from inspect import getmembers, isfunction, signature
from tqdm import tqdm
import numpy as np
import random
from shutil import move

sys.path.append('.')
from TSP_generator import *
import mhs

def reset_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

def dict_2_str_recurive(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    output = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            output += '%s%s%s%s' %(ident,braces*'[',key,braces*']')
            output += dict_2_str_recurive(value, ident+'  ', braces+1)
        else:
            output += ident+'%s = %s' %(key, value)

    return output

def framework_configurator():
    num_tsp_instances = input('Introduce the initial number of tsp instances for the experiment: (integer)\n')
    try:
        num_tsp_instances = int(num_tsp_instances)
    except ValueError:
        print('\'', num_tsp_instances, '\' could not be converted to integer', sep='')
        sys.exit(1)

    min_num_cities = input('Introduce the minimum number of cities for the TSP instances: (integer)\n')
    try:
        min_num_cities = int(min_num_cities)
    except ValueError:
        print('\'', min_num_cities, '\' could not be converted to integer', sep='')
        sys.exit(1)

    max_num_cities = input('Introduce the maximum number of cities for the TSP instances: (integer)\n')
    try:
        max_num_cities = int(max_num_cities)

        while max_num_cities <= min_num_cities:
            print('The maximum number of cities must be estrictly greater than the minimum number of cities')
            max_num_cities = input('Introduce the maximum number of cities for the TSP instances: (integer)\n')
            max_num_cities = int(max_num_cities)

    except ValueError:
        print('\'', max_num_cities, '\' could not be converted to integer', sep='')
        sys.exit(1)

    execution_max_time = input('Introduce the maximum running time of the MHs in seconds (600 seconds = 10 minutes): (integer)\n')
    try:
        execution_max_time = int(execution_max_time)
    except ValueError:
        print('\'', execution_max_time, '\' could not be converted to integer', sep='')
        sys.exit(1)

    framework_data = {}
    # framework_data['num_tsp_instances'] = num_tsp_instances
    # framework_data['min_num_cities'] = min_num_cities
    # framework_data['max_num_cities'] = max_num_cities
    # framework_data['execution_max_time'] = execution_max_time
    framework_data['TSP_generator'] = TSP_Experiment_Generator(num_tsp_instances, min_num_cities, max_num_cities, execution_max_time)
    hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()
    all_data = {'framework_data': framework_data, 'hash_string': hash_string}
    with open('data.pickle', 'wb') as f:
        pickle.dump(all_data, f)

def update_data_pickle(framework_data):
    hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()
    all_data = {'framework_data': framework_data, 'hash_string': hash_string}
    with open('data_new.pickle', 'wb') as f:
        pickle.dump(all_data, f)

    # os.rename('data_new.pickle', 'data.pickle')
    move('data_new.pickle', 'data.pickle')
    logging.info(str(datetime.now())+': DATABASE UPDATED: data.pickle')


def load_and_check_data_pickle():

    # BASIC LOAD AND CHECK CONSISTENCY
    if not path.exists('data.pickle'):
        print('Data file data.pickle not found. Running comparison framework configurator')
        framework_configurator()

    with open('data.pickle', 'rb') as f:
        all_data = pickle.load(f)
        framework_data = all_data['framework_data']
        read_hash_string = all_data['hash_string']
        real_hash_string = sha256(dict_2_str_recurive(framework_data).encode('utf-8')).hexdigest()

        if real_hash_string != read_hash_string:
            print('Corrupt file \'data.pickle\'. Please, remove it a run again the program')
            sys.exit(1)

    experimenter = framework_data['TSP_generator']
    print('Comparison framework with the following parameter settings:')
    print('Num of TSP instances:', len(experimenter.set_of_cities))
    print('Min number of cities:', experimenter.min_num_cities)
    print('Max number of cities:', experimenter.max_num_cities)
    print('Max MH run time:', experimenter.max_runtime)
    # print('TSP_generator:', framework_data['TSP_generator'])

    # CHECK THAT ALL MHs have been run on all the problems
    # experimenter = framework_data['TSP_generator']
    # algs = list(set([i for j in experimenter.set_of_cities for i in j.results]))
    #
    # for i in algs:
    #     for j in experimenter.set_of_cities:
    #         if i not in j.results:
    #             print('MH not runned:', i, j)
    #             sys.exit(1)

    return framework_data

def menu(data):
    experimenter = data['TSP_generator']
    algs = list(set([i for j in experimenter.set_of_cities for i in j.results]))
    results_allMHs_on_allInstances = True

    print('\nThere are results for:')
    for index, j in enumerate(experimenter.set_of_cities):
        print('Instance ', index, ': ', sep='', end='')
        for i_alg in algs:
            if i_alg in j.results:
                if len(j.results[i_alg]) >= 1:
                    print(i_alg, ', ', sep='', end='')
                # else:
                #     print('Error: Too little data for a MH. Please remove this data: ', i_alg, '(', len(j.results[i_alg]),')', sep='')
            else:
                results_allMHs_on_allInstances = False
        print('')

    print('\nMENU')
    print('1: Continue running MHs')
    print('2: Remove all the results of a MH')
    print('3: Increase the number of instances by one')
    print('4: Generate plots')
    print('^C: exit')
    try:
        option = input('Please, introduce your choice (1/2/3/4):\n')
    except KeyboardInterrupt:
        return 'exit'

    if option == '1':
        return option
    elif option == '2':
        mh_name = input('Please, introduce the name of the MH:\n')
        any_deletion = False
        confirmation = input('We are removing the results of ' + mh_name + '. Proceed? (y/n)\n')
        if confirmation == 'y':

            for index_instance, i_tsp_instance in enumerate(experimenter.set_of_cities):
                if mh_name in i_tsp_instance.results:
                    del i_tsp_instance.results[mh_name]
                    logging.info(str(datetime.now())+': RESULTS REMOVED (pickle update pending): '+mh_name+' on TSP instance '+str(index_instance))
                    any_deletion = True
            if not any_deletion:
                print('No results found (nor removed)')
            else:
                update_data_pickle(data)
                print('Deletion completed')
        else:
            print('Deletion cancelled')

    elif option == '3':
        experimenter.introduce_a_new_instance()
        logging.info(str(datetime.now()) + ': NEW TSP INSTANCE (pickle update pending): ' + str(len(experimenter.set_of_cities) - 1))
        update_data_pickle(data)
        print('Instance created and added')
    elif option == '4':
        if not results_allMHs_on_allInstances:
            print('Plots not generated. All the MHs must have been applied on all the TSP instance.')
        else:
            print('Generating plots:')
            with tqdm(total=6) as pbar:
                data['TSP_generator'].plot_convergence_graphs('convergence_graphs_xlog_ylog.png', True, True)
                pbar.update()
                data['TSP_generator'].plot_convergence_graphs('convergence_graphs_ylog.png', False, True)
                pbar.update()
                data['TSP_generator'].plot_convergence_graphs('convergence_graphs.png', False, False)
                pbar.update()
                data['TSP_generator'].plot_rank_evolution_graph('rank_evolution.png', False, False)
                pbar.update()
                data['TSP_generator'].plot_convergence_graphs('convergence_graphs_xlog.png', True, False)
                pbar.update()
                data['TSP_generator'].plot_rank_evolution_graph('rank_evolution_xlog.png', True, False)
                pbar.update()
    else:
        print('Not a valid option: ', option)

    return None


def run(mh_function, tsp_instance, index):

    if mh_function.__name__ not in tsp_instance.results:
        def get_fitness_function(name):
            def fitness_function(solution):
                return tsp_instance.evaluate_and_resgister(solution, name)
            return fitness_function
        # mh_function(i.size(), i.evaluate_and_resgister)
        ffunction = get_fitness_function(mh_function.__name__)
        num_cities = tsp_instance.size()
        try:
            logging.info(str(datetime.now())+': EXECUTION BEGINS: ' + mh_function.__name__ + ' on TSP intsance ' + str(index))
            mh_function(num_cities, ffunction)
        except MaxRuntimeHit:
            logging.info(str(datetime.now())+': EXECUTION ENDED (pickle update pending): ' + mh_function.__name__ + ' on TSP intsance ' + str(index))
            update_data_pickle(data)
        except KeyboardInterrupt:
            print('Executions interrupted')
            logging.info(str(datetime.now())+': EXECUTION INTERRUPTED: ' + mh_function.__name__ + ' on TSP intsance ' + str(index))
            raise
        except Exception as e:
            print('ERROR: Something went wrong with a MH. See log.log', mh_function.__name__)
            logging.error(str(datetime.now())+':\n'+str(e), exc_info=True)
            sys.exit(1)
        else:
            print('No MAX_RUNTIME_HIT: The following algorithm terminated before the maximal runtime and results were not saved: ' + mh_function.__name__)
            print('Did you replace the main loop with While True:?')
            sys.exit(1)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, filename='log.log', filemode='a')
    data = load_and_check_data_pickle()
    experimenter = data['TSP_generator']

    while True:
        try:
            option = menu(data)
        except KeyboardInterrupt:
            option = 'None'

        if option == '1':

            # Get the list of executions not runned
            pending_executions = []
            for name, mh_function in getmembers(mhs, isfunction):
                sig = signature(mh_function)
                if len(sig.parameters) == 2:
                    for index, instance in enumerate(experimenter.set_of_cities):
                        if name not in instance.results:
                            pending_executions.append((mh_function, instance, index))
                else:
                    print('There are functions in mhs.py with less or more than two arguments. This is discouraged! ', name+str(sig))
                    print(
                        'mhs.py should just contains a function per MH and every one should recieve exactly two arguments: ', end='')
                    print('(N, f), with N the size of the problem and f the evaluation function')
                    print('In case some operators of the MH are defined as another function, include them into the function of the MH. For instance:')
                    print('def simulated_annealing(N, f):')
                    print('   def neigh_operator(solution):')
                    print('      ...')
                    print('   ...')
                    print('   new_solution = neigh_operator(solution)')
                    print('   ...\n')
                    logging.warning(str(datetime.now()) + ': NO MH in mhs.py: ' + name + str(sig))

            print('EXECUTING MHs (you can kill the process with ^C whenever you want. Results of the executions would be stored incrementally. Those of the last execution would be discarded)')
            for i in tqdm(pending_executions):
                try:
                    run(i[0], i[1], i[2])
                except KeyboardInterrupt:
                    option = 'exit'
                    break

        if option == 'exit':
            break

        _ = input('Press Enter to continue')
        sys.stdin.flush()
