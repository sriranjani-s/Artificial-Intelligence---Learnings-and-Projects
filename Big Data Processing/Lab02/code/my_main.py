
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

import create_benchmark
import solve_benchmark
import sys
import codecs

# ------------------------------------------
# FUNCTION get_name
# ------------------------------------------
def get_report_name(population_size, num_movies, use_fast_algorithm, num_cores):
    # 1. We create the output variable
    res = ""

    # 2. We get the sequential or parallel flavour
    if (num_cores == 1):
        res = res + "Sequential_"
    else:
        res = res + "Parallel-" + str(num_cores) + "-Cores_"

    # 3. We get the type of algorithm
    if (use_fast_algorithm == True):
        res = res + "Algorithm-nlogn_"
    else:
        res = res + "Algorithm-n2_"

    # 4. We get the num of files and size of each of them
    res = res + "People-" + str(population_size) + "_Movies-" + str(num_movies) + ".txt"

    # 5. We return res
    return res

# ------------------------------------------
# FUNCTION write_report
# ------------------------------------------
def write_report(output_file_name, population_results, total_time, time_per_core, num_cores):
    # 1. We open the file for writing
    my_output_file = codecs.open(output_file_name, "w", encoding='utf-8')

    # 2. We print the results per file
    my_output_file.write("\n--------------------\n FILE RESULTS\n--------------------\n")
    for item in population_results:
        my_output_file.write(item[0] + " --> " + str(item[1]) + " seconds, " + str(item[2]) + " inversions\n")

    # 3. We print the total time
    my_output_file.write("\n--------------------\n TIMES\n--------------------\n")
    my_output_file.write("TOTAL TIME = " + str(total_time) + " seconds\n\n")

    # 4. We print the concurrency time
    my_output_file.write("CONCURRENCY TIME = " + str(max(time_per_core)) + " seconds\n")

    # 5. We write the times per core
    for index in range(num_cores):
        my_output_file.write("Time Core " + str(index) + " = " + str(time_per_core[index]) + " seconds\n")

    # 6. We close the file
    my_output_file.close()

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(new_benchmark, population_size, num_movies, use_fast_algorithm, num_cores):
    # 1. If a new benchmark is required we generate it
    if (new_benchmark == True):
        create_benchmark.generate_benchmark("../input_files/", population_size, num_movies)

    # 2. We get the name of the result file
    report_file_name = get_report_name(population_size, num_movies, use_fast_algorithm, num_cores)

    # 3. We solve the benchmark under the required parameters
    (population_results, total_time, time_per_core) = solve_benchmark.run_benchmark("../input_files/", use_fast_algorithm, num_cores)

    # 4. We write the report
    write_report("../results/" + report_file_name, population_results, total_time, time_per_core, num_cores)

# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We collect the input values

    # 1.1. If we call the program from the console then we collect the arguments from it
    if (len(sys.argv) > 1):
        my_arg1 = sys.argv[1]
        if (my_arg1 == "True"):
            new_benchmark = True
        else:
            new_benchmark = False

        population_size = int(sys.argv[2])
        num_movies = int(sys.argv[3])

        my_arg4 = sys.argv[4]
        if (my_arg4 == "True"):
            use_fast_algorithm = True
        else:
            use_fast_algorithm = False

        num_cores = int(sys.argv[5])

    # 1.2. If we call the program from PyCharm then we hardcode the arguments to the values we want
    else:
        new_benchmark = True
        population_size = 3
        num_movies = 8
        use_fast_algorithm = True
        num_cores = 2

    # 2. We call to my_main
    my_main(new_benchmark, population_size, num_movies, use_fast_algorithm, num_cores)
