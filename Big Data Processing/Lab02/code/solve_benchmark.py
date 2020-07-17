
# --------------------------------------------------------
#           PYTHON PROGRAM
# Here is where we are going to define our set of...
# - Imports
# - Global Variables
# - Functions
# ...to achieve the functionality required.
# When executing > python 'this_file'.py in a terminal,
# the Python interpreter will load our program,
# but it will execute nothing yet.
# --------------------------------------------------------

import time
import os
import codecs
import math
import multiprocessing
import problem_algorithms

# ------------------------------------------
# FUNCTION run_file
# ------------------------------------------
def run_file(file_name, use_fast_algorithm):
    # 1. We create the variable to return
    res = ()

    # 1.1. We output the elapsed time
    elapsed_time = 0

    # 1.2. We output the number of inversions
    num_inv = 0

    # 2. We read the file content to an input list

    # 2.1. We open the file for reading
    my_input_file = codecs.open(file_name, "r", encoding='utf-8')

    # 2.2. We read it
    movie_preferences = [ int(line.strip()) for line in my_input_file ]

    # 2.3. We close the file
    my_input_file.close()

    # 3. We solve the problem

    # 3.1. We check the current time
    start_time = time.time()

    # 3.2. We solve the problem using the desired algorithm
    if (use_fast_algorithm == True):
        num_inv = problem_algorithms.count_inversions_nlogn(movie_preferences)
    else:
        num_inv = problem_algorithms.count_inversions_n2(movie_preferences)

    # 3.3. We check the current time again to measure the elapsed time on solving the problem
    elapsed_time = time.time() - start_time

    # 4. We assign res
    res = (file_name, elapsed_time, num_inv)

    # 5. We return res
    return res

# --------------------------------------------------
# my_divide_stage
# --------------------------------------------------
def my_divide_stage(population_files, num_cores, use_fast_algorithm):
    # 1. We create the output variable
    res = []

    # 2. We get the amount of people in the population benchmark
    size = len(population_files)

    # 3. We get the size of each people_subset to be assigned to a different core
    sub_size = math.ceil((size * 1.0) / (num_cores * 1.0))

    # 4. We assign res
    res = [ (population_files[slice_lb:(slice_lb + sub_size)], use_fast_algorithm) for slice_lb in range(0, size, sub_size)]

    # 5. We return res
    return res

#--------------------------------------------------
# FUNCTION core_workload
#--------------------------------------------------
def core_workload(slice):
    # 1. We create the output variable
    res = []

    # 2. We unpack the variables
    population = slice[0]
    use_fast_algorithm = slice[1]

    # 3. We assign res
    res = [ run_file(person, use_fast_algorithm) for person in population ]

    # 4. We return res
    return res

#--------------------------------------------------
# FUNCTION my_map_stage
#--------------------------------------------------
def my_map_stage(population_slices):
    # 1. We create the output variable
    res = []

    # 2. We setup the object for enabling parallel computation among the different cores
    pool = multiprocessing.Pool()

    # 3. We use pool to trigger the parallel execution of each people subset (slice) in a different process
    res = pool.map( core_workload, population_slices )

    # 4. We return res
    return res

#--------------------------------------------------
# FUNCTION my_reduce_stage
#--------------------------------------------------
def my_reduce_stage(population_slices_results):
    # 1. We create the output variable
    res = ()

    # 1.1. We output the inversions and elapsed time per person in the population
    population_results = [ person for slice in population_slices_results for person in slice ]

    # 1.2. Additionally, we output as well the total working time per core (which is relevant to our analysis)
    time_per_core = [ sum(person[1] for person in slice) for slice in population_slices_results ]

    # 1.3. Additionally, we output the total time spent by all cores all together
    total_time = sum(time_per_core)

    # 2. We assign res to its final value
    res = (population_results, total_time, time_per_core)

    # 3. We return res
    return res

# ------------------------------------------
# FUNCTION run_benchmark
# ------------------------------------------
def run_benchmark(input_files_dir, use_fast_algorithm, num_cores):
    # 1. We create the output variable
    res = ()

    # 1.1. We output the population results
    population_results = []

    # 1.2. We output the total time
    total_time = 0

    # 1.3. We output the time per core
    time_per_core = []

    # 2. We get the list of files from input_files in lexicographic order
    population_files = [ input_files_dir + file_name for file_name in os.listdir(input_files_dir) ]
    population_files.sort()

    # 3. If we follow sequential mode...
    if (num_cores == 1):
        # 3.1. We solve all the person files for the population, one after another
        population_results = [ run_file(person, use_fast_algorithm)  for person in population_files ]

        # 3.2. We compute the total time
        total_time = sum(person_results[1] for person_results in population_results)

        # 3.3. As all the workload has been done by a single core, we assign the total time to it
        time_per_core = [ total_time ]

    # 4. If we follow parallel mode...
    else:
        # 4.1. We apply our problem-specific divide function:
        population_slices = my_divide_stage(population_files, num_cores, use_fast_algorithm)

        # 4.2. We apply our problem-specific map function:
        population_slices_results = my_map_stage(population_slices)

        # 4.3. We apply our problem-specific reduce function:
        (population_results, total_time, time_per_core) = my_reduce_stage(population_slices_results)

    # 5. We assign res
    res = (population_results, total_time, time_per_core)

    # 6. We return res
    return res
