# Databricks notebook source
# --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------

import pyspark
from datetime import datetime

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

    # ----------
    # Schema:
    # ----------
    # 0 -> station_number
    # 1 -> station_name
    # 2 -> direction
    # 3 -> day_of_week
    # 4 -> date
    # 5 -> query_time
    # 6 -> scheduled_time
    # 7 -> expected_arrival_time

    if (len(params) == 8):
        res = (int(params[0]),
               str(params[1]),
               str(params[2]),
               str(params[3]),
               str(params[4]),
               str(params[5]),
               str(params[6]),
               str(params[7])
               )

    # 5. We return res
    return res

# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(sc, my_dataset_dir):
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We count the total amount of entries
    resVal = inputRDD.count()

    # 3. We print the result
    print(resVal)

# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(sc, my_dataset_dir, station_number):
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
   
    # 3. We filter records for the required station number
    BusStationRDD = allLinesRDD.filter(lambda x : x[0] == station_number)

    # 4. We retrieve the distinct calender days
    CalenderDatesRDD = BusStationRDD.map(lambda x : x[4]).distinct()
    
    # 5. We count the total amount of entries
    resVal = CalenderDatesRDD.count()
    
    # 6. We print the result
    print(resVal) 

# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(sc, my_dataset_dir, station_number):
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
   
    # 3. We filter records for the required station number
    BusStationRDD = allLinesRDD.filter(lambda x : x[0] == station_number)
    
    # 4. We persist the data
    BusStationRDD.persist()
    
    # 5. We count the entries for 1st condition
    Accum1 = BusStationRDD.filter(lambda x: (x[6] >= x[7])).count()
    
    # 6. We count the entries for 2nd condition
    Accum2 = BusStationRDD.filter(lambda x: (x[6] < x[7])).count()
    
    # 7. We print the results
    print((Accum1,Accum2))


# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(sc, my_dataset_dir, station_number):
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
   
    # 3. We filter records for the required station number
    BusStationRDD = allLinesRDD.filter(lambda x : x[0] == station_number)
    
    # 4. We extract distinct scheduled time values and sort it.
    DaysRDD = BusStationRDD.map(lambda x: (x[3], x[6])).distinct().sortBy(lambda x: x[1])
    
    # 5. We combine the values as a list for the key
    combineRDD = DaysRDD.combineByKey(lambda x:[x],lambda x,y:x+[y],lambda x,y:x+y)
    
    # 6. We collect the results
    resVal = combineRDD.collect()
    
    # 7. We print the results
    for item in resVal:
      print(item)

# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(sc, my_dataset_dir, station_number, month_list):
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir) 

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
   
    # 3. We filter records for the required station number and the months
    BusStationRDD = allLinesRDD.filter(lambda x : x[0] == station_number and x[4][3:5] in month_list)
    
    # 4. We compute the waiting time difference
    WaitingTimesRDD = BusStationRDD.map(lambda x: (x[3] + ' ' + x[4][3:5], abs((datetime.strptime(x[7], "%H:%M:%S")- datetime.strptime(x[5], "%H:%M:%S")).seconds)))
    
    # 5. We aggregate the values
    AccumRDD = WaitingTimesRDD.combineByKey(lambda val: (val, 1),
                                 lambda accum, new_val: (accum[0] + new_val, accum[1] + 1),
                                 lambda final_accum1, final_accum2: (final_accum1[0] + final_accum2[0], final_accum1[1] + final_accum2[1])
                                 )
    
    # 6. We find the average
    AverageRDD = AccumRDD.mapValues(lambda x: x[0] / x[1])

    # 7. We sort and collect the results
    resVal = AverageRDD.sortBy(lambda x: x[1]).collect()
   
    # 8. We print the results 
    for item in resVal:
      print(item)

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, option):
    # Exercise 1:
    # Number of measurements per station number
    if option == 1:
        ex1(sc, my_dataset_dir)

    # Exercise 2: Station 240101 (UCC WGB - Lotabeg):
    # Number of different days for which data is collected.
    if option == 2:
        ex2(sc, my_dataset_dir, 240101)

    # Exercise 3: Station 240561 (UCC WGB - Curraheen):
    # Number of buses arriving ahead and behind schedule.
    if option == 3:
        ex3(sc, my_dataset_dir, 240561)

    # Exercise 4: Station 241111 (CIT Technology Park - Lotabeg):
    # List of buses scheduled per day of the week.
    if option == 4:
        ex4(sc, my_dataset_dir, 241111)

    # Exercise 5: Station 240491 (Patrick Street - Curraheen):
    # Average waiting time per day of the week during the Semester1 months. Sort the entries by decreasing average waiting time.
    if option == 5:
        ex5(sc, my_dataset_dir, 240491, ['09','10','11'])

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

    my_dataset_dir = "Filestore/tables/7_Assignments/A02/my_dataset_single_file/"

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