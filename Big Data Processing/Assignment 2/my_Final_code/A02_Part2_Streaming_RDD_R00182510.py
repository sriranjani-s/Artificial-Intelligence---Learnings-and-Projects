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
import pyspark.streaming

import os
import shutil
import time
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


def my_functionex3(times):
  
    Accum1 = 0
    Accum2 = 0
    
    if times[0] >= times[1]:
      Accum1 += 1
    else:
      Accum2 += 1
    
    return(Accum1,Accum2)
    
# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(ssc, monitoring_dir):
    # 1. Operation C1: 'textFileStream'
    inputDStream = ssc.textFileStream(monitoring_dir)

    # 2. Count the records
    solutionDStream = inputDStream.count()
    
    # 3. Print the results
    solutionDStream.pprint()

# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(ssc, monitoring_dir, station_number):
    # 1. Operation C1: 'textFileStream'
    inputDStream = ssc.textFileStream(monitoring_dir)

    # 2. Format input records
    alllinesDStream = inputDStream.map(lambda x: process_line(x))
    
    # 3. We filter records for the required station number
    FilteredDStream = alllinesDStream.filter(lambda x : x[0] == station_number)
    
    # 3. We retrieve the unique calender days
    AllDaysDStream = FilteredDStream.map(lambda x: (x[4], 1))
    
    # 4. We count the total amount of entries
    solutionDStream = AllDaysDStream.reduceByKey(lambda x, y: x + y).count()
    
    # 5. We print the result
    solutionDStream.pprint()
    
# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(ssc, monitoring_dir, station_number):
    # 1. Operation C1: 'textFileStream'
    inputDStream = ssc.textFileStream(monitoring_dir)
    
    # 2. We process each line to get the relevant info
    alllinesDStream = inputDStream.map(lambda x: process_line(x))
    
    # 3. We filter records for the required station number
    FilteredDStream = alllinesDStream.filter(lambda x: x[0] == station_number)
    
    # 4. We map to a user defined function to retrieve a tuple of computed values
    CountDStream = FilteredDStream.map(lambda x : my_functionex3((x[6], x[7])))
    
    # 5. We aggregate the result
    SolutionDStream = CountDStream.reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
    
    # 6. We print the result
    SolutionDStream.pprint()

# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(ssc, monitoring_dir, station_number):
    # 1. Operation C1: 'textFileStream'
    inputDStream = ssc.textFileStream(monitoring_dir)
    
    # 2. We process each line to get the relevant info
    alllinesDStream = inputDStream.map(lambda x: process_line(x))
    
    # 3. We filter records for the required station number
    FilteredDStream = alllinesDStream.filter(lambda x : x[0] == station_number)
    
    # 4. We extract the scheduled time values
    AllDaysDStream = FilteredDStream.map(lambda x: ((x[3], x[6]), 1))
    
    # 5. We combine the values for a key
    AccumDStream = AllDaysDStream.reduceByKey(lambda x, y: x + y).map(lambda x: x[0])
    
    # 6. We group the values for a key as a list
    SolutionDStream = AccumDStream.combineByKey(lambda x:[x],lambda x,y:x+[y],lambda x,y:x+y)
    
    # 7. We print the result
    SolutionDStream.pprint()

# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(ssc, monitoring_dir, station_number, month_list):
    # 1. Operation C1: 'textFileStream'
    inputDStream = ssc.textFileStream(monitoring_dir)
    
    # 2. We process each line to get the relevant info
    alllinesDStream = inputDStream.map(lambda x: process_line(x))
   
    # 3. We filter records for the required station number and the months
    FilteredDStream = alllinesDStream.filter(lambda x : x[0] == station_number and x[4][3:5] in month_list)
    
    # 4. We compute the waiting time difference
    WaittimeDStream = FilteredDStream.map(lambda x: (x[3] + ' ' + x[4][3:5]\
                                     ,(datetime.strptime(x[7], "%H:%M:%S").timestamp() - datetime.strptime(x[5], "%H:%M:%S").timestamp())))
    
    # 5. We aggregate the values
    AccumDStream = WaittimeDStream.combineByKey(lambda val: (val, 1),
                                 lambda accum, new_val: (accum[0] + new_val, accum[1] + 1),
                                 lambda final_accum1, final_accum2: (final_accum1[0] + final_accum2[0], final_accum1[1] + final_accum2[1])
                                 )
    
    # 6. We find the average
    AverageDStream = AccumDStream.mapValues(lambda x: x[0] / x[1])
    
    # 7. We sort the results using transform operation
    SolutionDStream = AverageDStream.transform( lambda rdd: rdd.sortBy(lambda item: item[1]) )
    
    # 8. We print the results
    SolutionDStream.pprint()


# ------------------------------------------
# FUNCTION my_model
# ------------------------------------------
def my_model(ssc, monitoring_dir, option):
    # Exercise 1:
    # Number of measurements per station number
    if option == 1:
        ex1(ssc, monitoring_dir)

    # Exercise 2: Station 240101 (UCC WGB - Lotabeg):
    # Number of different days for which data is collected.
    if option == 2:
        ex2(ssc, monitoring_dir, 240101)

    # Exercise 3: Station 240561 (UCC WGB - Curraheen):
    # Number of buses arriving ahead and behind schedule.
    if option == 3:
        ex3(ssc, monitoring_dir, 240561)

    # Exercise 4: Station 241111 (CIT Technology Park - Lotabeg):
    # List of buses scheduled per day of the week.
    if option == 4:
        ex4(ssc, monitoring_dir, 241111)

    # Exercise 5: Station 240491 (Patrick Street - Curraheen):
    # Average waiting time per day of the week during the Semester1 months. Sort the entries by decreasing average waiting time.
    if option == 5:
        ex5(ssc, monitoring_dir, 240491, ['09','10','11'])


# ------------------------------------------
# FUNCTION get_source_dir_file_names
# ------------------------------------------
def get_source_dir_file_names(local_False_databricks_True, source_dir, verbose):
    # 1. We create the output variable
    res = []

    # 2. We get the FileInfo representation of the files of source_dir
    fileInfo_objects = []
    if local_False_databricks_True == False:
        fileInfo_objects = os.listdir(source_dir)
    else:
        fileInfo_objects = dbutils.fs.ls(source_dir)

    # 3. We traverse the fileInfo objects, to get the name of each file
    for item in fileInfo_objects:
        # 3.1. We get a string representation of the fileInfo
        file_name = str(item)

        # 3.2. If the file is processed in DBFS
        if local_False_databricks_True == True:
            # 3.2.1. We look for the pattern name= to remove all useless info from the start
            lb_index = file_name.index("name='")
            file_name = file_name[(lb_index + 6):]

            # 3.2.2. We look for the pattern ') to remove all useless info from the end
            ub_index = file_name.index("',")
            file_name = file_name[:ub_index]

        # 3.3. We append the name to the list
        res.append(file_name)
        if verbose == True:
            print(file_name)

    # 4. We sort the list in alphabetic order
    res.sort()

    # 5. We return res
    return res


# ------------------------------------------
# FUNCTION streaming_simulation
# ------------------------------------------
def streaming_simulation(local_False_databricks_True,
                         source_dir,
                         monitoring_dir,
                         time_step_interval,
                         verbose,
                         dataset_file_names
                        ):

    # 1. We check what time is it
    start = time.time()

    # 2. We set a counter in the amount of files being transferred
    count = 0

    # 3. If verbose mode, we inform of the starting time
    if (verbose == True):
        print("Start time = " + str(start))

    # 4. We transfer the files to simulate their streaming arrival.
    for file in dataset_file_names:
        # 4.1. We copy the file from source_dir to dataset_dir
        if local_False_databricks_True == False:
            shutil.copyfile(source_dir + file, monitoring_dir + file)
        else:
            dbutils.fs.cp(source_dir + file, monitoring_dir + file)

        # 4.2. If verbose mode, we inform from such transferrence and the current time.
        if (verbose == True):
            print("File " + str(count) + " transferred. Time since start = " + str(time.time() - start))

        # 4.3. We increase the counter, as we have transferred a new file
        count = count + 1

        # 4.4. We wait the desired transfer_interval until next time slot.
        time_to_wait = (start + (count * time_step_interval)) - time.time()
        if (time_to_wait > 0):
            time.sleep(time_to_wait)


# ------------------------------------------
# FUNCTION create_ssc
# ------------------------------------------
def create_ssc(sc, monitoring_dir, time_step_interval, option):
    # 1. We create the new Spark Streaming context acting every time_step_interval.
    ssc = pyspark.streaming.StreamingContext(sc, time_step_interval)

    # 2. We model the data processing to be done each time_step_interval.
    my_model(ssc, monitoring_dir, option)

    # 3. We return the ssc configured and modelled.
    return ssc


# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            time_step_interval,
            verbose,
            option
           ):

    # 1. We get the names of the files of our dataset
    dataset_file_names = get_source_dir_file_names(local_False_databricks_True, source_dir, verbose)

    # 2. We setup the Spark Streaming context.
    # This sets up the computation that will be done when the system receives data.
    ssc = pyspark.streaming.StreamingContext.getActiveOrCreate(checkpoint_dir,
                                                               lambda: create_ssc(sc,
                                                                                  monitoring_dir,
                                                                                  time_step_interval,
                                                                                  option
                                                                                 )
                                                               )

    # 3. We start the Spark Streaming Context in the background to start receiving data.
    #    Spark Streaming will start scheduling Spark jobs in a separate thread.
    ssc.start()
    ssc.awaitTerminationOrTimeout(time_step_interval)

    # 4. We simulate the streaming arrival of files (i.e., one by one) from source_dir to monitoring_dir.
    streaming_simulation(local_False_databricks_True,
                         source_dir,
                         monitoring_dir,
                         time_step_interval,
                         verbose,
                         dataset_file_names
                        )

    # 5. We stop the Spark Streaming Context
    ssc.stop(False)
    if (not sc._jvm.StreamingContext.getActive().isEmpty()):
        sc._jvm.StreamingContext.getActive().get().stop(False)


# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed

    # 1.1. We specify the time interval each of our micro-batches (files) appear for its processing.
    time_step_interval = 5

    # 1.2. We configure verbosity during the program run
    verbose = False

    # 1.3. We specify the exercise we want to solve
    option = 5

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset, my_monitoring, my_checkpoint and my_result
    my_local_path = "/home/nacho/CIT/Tools/MyCode/Spark/"
    my_databricks_path = "/"

    source_dir = "Filestore/tables/7_Assignments/A02/my_dataset_complete/"
    monitoring_dir = "Filestore/tables/7_Assignments/A02/my_monitoring/"
    checkpoint_dir = "Filestore/tables/7_Assignments/A02/my_checkpoint/"

    if local_False_databricks_True == False:
        source_dir = my_local_path + source_dir
        monitoring_dir = my_local_path + monitoring_dir
        checkpoint_dir = my_local_path + checkpoint_dir
    else:
        source_dir = my_databricks_path + source_dir
        monitoring_dir = my_databricks_path + monitoring_dir
        checkpoint_dir = my_databricks_path + checkpoint_dir

    # 4. We remove the directories
    if local_False_databricks_True == False:
        # 4.1. We remove the monitoring_dir
        if os.path.exists(monitoring_dir):
            shutil.rmtree(monitoring_dir)

        # 4.2. We remove the checkpoint_dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    else:
        # 4.1. We remove the monitoring_dir
        dbutils.fs.rm(monitoring_dir, True)

        # 4.2. We remove the checkpoint_dir
        dbutils.fs.rm(checkpoint_dir, True)

    # 5. We re-create the directories again
    if local_False_databricks_True == False:
        # 5.1. We re-create the monitoring_dir
        os.mkdir(monitoring_dir)

        # 5.2. We re-create the checkpoint_dir
        os.mkdir(checkpoint_dir)
    else:
        # 5.1. We re-create the monitoring_dir
        dbutils.fs.mkdirs(monitoring_dir)

        # 5.2. We re-create the checkpoint_dir
        dbutils.fs.mkdirs(checkpoint_dir)

    # 6. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 7. We call to our main function
    my_main(sc,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            time_step_interval,
            verbose,
            option
           )