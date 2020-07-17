# Databricks notebook source
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

import pyspark

# ------------------------------------------
# FUNCTION process_line
# ------------------------------------------
def process_line(line):
    # 1. We create the output variable
    res = ()

    # 2. We remove the end of line character
    line = line.replace("\n", "")

    # 3. We split the line by tabulator characters
    params = line.split(";")

    # 4. We assign res
    if (len(params) == 7):
        res = tuple(params)

    # 5. We return res
    return res


# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(sc, my_dataset_dir):
    pass

    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We count the total amount of entries
    resVal = inputRDD.count()

    # 3. We print the result
    print(resVal)


# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(sc, my_dataset_dir):
    pass

    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)

    # 3. We project just the info we are interested into
    StationsRDD = allLinesRDD.map(lambda x: x[1])

    # 4. We get just the entries that are different
    StationDistinctRDD = StationsRDD.distinct()

    # 5. We count such these entries
    Station_count = StationDistinctRDD.count()

    # 6. We print the result
    print(Station_count) 


# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(sc, my_dataset_dir):
    pass

    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)

    # 3. We project just the info we are interested into
    StationsRDD = allLinesRDD.map(lambda x: x[1])

    # 4. We get just the entries that are different
    StationDistinctRDD = StationsRDD.distinct()

    # 5. We collect such these entries
    resVal = StationDistinctRDD.collect()

    # 6. We print them
    for item in resVal:
      print(item)


# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(sc, my_dataset_dir):
    pass

    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)

    # 3. We project just the info we are interested into
    StationlongitudeRDD = allLinesRDD.map(lambda x: (x[2],x[1]))

    # 4. We get just the entries that are different
    LongitudeRDD = StationlongitudeRDD.distinct()

    # 5. We sort them by their longitude
    sortedLongitudeRDD = LongitudeRDD.sortByKey()

    # 6. We collect such these entries
    resVal = sortedLongitudeRDD.collect()

    # 7. We print them
    for (key,value) in resVal:
      print((value,key))

# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(sc, my_dataset_dir):
    pass

    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)

    # 3. We filter the bikes of "Kent Station"
    KentStationRDD = allLinesRDD.filter(lambda x : x[1] == 'Kent Station')

    # 4. We project just the info we are interested into
    KentBikesRDD = KentStationRDD.map(lambda x : int(x[5]))

    # 5. We compute the average amount of bikes
    resVAL = KentBikesRDD.aggregate((0, 0),
                                    lambda acc, e: (acc[0] + e, acc[1] + 1),
                                    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
                                    )

    # 6. We print this info by the screen
    print("TotalSum = " + str(resVAL[0]))
    print("TotalItems = " + str(resVAL[1]))
    print("AverageValue = " + str((resVAL[0] * 1.0) / (resVAL[1] * 1.0)))

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, option):
    # Exercise 1: Total amount of entries in the dataset.
    if option == 1:
        ex1(sc, my_dataset_dir)

    # Exercise 2: Number of Coca-cola bikes stations in Cork.
    if option == 2:
        ex2(sc, my_dataset_dir)

    # Exercise 3: List of Coca-Cola bike stations.
    if option == 3:
        ex3(sc, my_dataset_dir)

    # Exercise 4: Sort the bike stations by their longitude (East to West).
    if option == 4:
        ex4(sc, my_dataset_dir)

    # Exercise 5: Average number of bikes available at Kent Station.
    if option == 5:
        ex5(sc, my_dataset_dir)

# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    option = 5

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = "/home/nacho/CIT/Tools/MyCode/Spark/"
    my_databricks_path = "/"

    my_dataset_dir = "Filestore/tables/7_Assignments/A01/my_dataset/"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, option)
