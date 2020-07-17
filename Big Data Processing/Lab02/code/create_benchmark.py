
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

import codecs
import random
import os
import shutil

# ------------------------------------------
# FUNCTION generate_file
# ------------------------------------------
def generate_file(num_movies, file_name):
    # 1. We create the list
    num_list = [ (value + 1) for value in range(num_movies) ]

    # 2. We open the file to write
    my_input_file = codecs.open(file_name, "w", encoding='utf-8')

    # 3. We fill the file with the list content
    for iteration in range(num_movies):
        # 3.1. We pick a element from the list
        index = random.randint(0, num_movies-1)

        # 3.2. We print the item
        my_input_file.write(str(num_list[index]) + "\n")

        # 3.3. We delete the item
        del num_list[index]

        # 3.4. We decrease the number of movies
        num_movies = num_movies - 1

    # 4. We close the file
    my_input_file.close()

# ------------------------------------------
# FUNCTION generate_benchmark
# ------------------------------------------
def generate_benchmark(directory_name, num_people, num_movies):
    # 1. If the directory already contained some files, we remove them
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
        os.mkdir(directory_name)

    # 2. We generate the benchmark by creating the desired number of files
    for index in range(num_people):
        generate_file(num_movies, directory_name + "file_" + str(index + 1) + ".txt")
