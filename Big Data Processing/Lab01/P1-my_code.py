import os
import numpy


# ------------------------------------------
# FUNCTION parse_in
# ------------------------------------------
def parse_in(input_name):
    with open(input_name) as f:
        row, col = [int(x) for x in f.readline().split()]
        my_data = []
        for line in f:
            my_data.append([x for x in line.split()])
    return (row, col, my_data)


# ------------------------------------------
# FUNCTION solve
# ------------------------------------------
def solve(my_data):
    row, col, mine_array = my_data
    for i in range(row):
        for j in range(col):
            if mine_array[i][j] != 'x':
                if i == 0 and j == 0:
                    neighbours = [mine_array[i][j + 1], mine_array[i + 1][j], mine_array[i + 1][j + 1]]
                elif i == 0 and j == col - 1:
                    neighbours = [mine_array[i][j - 1], mine_array[i + 1][j - 1], mine_array[i + 1][j]]
                elif i == row - 1 and j == col - 1:
                    neighbours = [mine_array[i - 1][j], mine_array[i - 1][j - 1], mine_array[i][j - 1]]
                elif i == row - 1 and j == 0:
                    neighbours = [mine_array[i - 1][j], mine_array[i - 1][j + 1], mine_array[i][j + 1]]
                elif i > 0 and j == 0:
                    neighbours = [mine_array[i - 1][j], mine_array[i - 1][j + 1], mine_array[i][j + 1],
                                  mine_array[i + 1][j + 1], mine_array[i + 1][j]]
                elif i == 0 and j > 0:
                    neighbours = [mine_array[i][j - 1], mine_array[i + 1][j - 1], mine_array[i + 1][j],
                                  mine_array[i + 1][j + 1], mine_array[i][j + 1]]
                elif i > 0 and j == col - 1:
                    neighbours = [mine_array[i - 1][j], mine_array[i - 1][j - 1], mine_array[i][j - 1],
                                  mine_array[i + 1][j - 1], mine_array[i + 1][j]]
                elif i == row - 1 and j > 0:
                    neighbours = [mine_array[i][j - 1], mine_array[i - 1][j - 1], mine_array[i - 1][j],
                                  mine_array[i - 1][j + 1], mine_array[i][j + 1]]
                else:
                    neighbours = [mine_array[i - 1][j - 1], mine_array[i - 1][j], mine_array[i - 1][j + 1],
                                  mine_array[i][j - 1], mine_array[i][j + 1], mine_array[i + 1][j - 1],
                                  mine_array[i + 1][j], mine_array[i + 1][j + 1]]
                mine_array[i][j] = neighbours.count('x')
    return (mine_array)


# ------------------------------------------
# FUNCTION parse_out
# ------------------------------------------
def parse_out(output_name, my_solution):
    print(my_solution)
    with open(output_name, 'w') as f:
        for elem in my_solution:
            f.write(" ".join(str(x) for x in elem))
            f.write("\n")


# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(input_name, output_name):
    # 1. We do the parseIn from the input file
    my_data = parse_in(input_name)

    # 2. We do the strategy to solve the problem
    my_solution = solve(my_data)

    # 3. We do the parse out to the output file
    parse_out(output_name, my_solution)


os.getcwd()
# 1. Name of input and output files
input_name = "input_1.txt"
output_name = "outputt.txt"

# 2. Main function
my_main(input_name, output_name)

