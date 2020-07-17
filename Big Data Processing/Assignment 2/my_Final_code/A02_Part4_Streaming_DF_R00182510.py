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
import pyspark.sql.functions as F

import os
import shutil
import time

# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval):
    
    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("station_number", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("station_name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("direction", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("day_of_week", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("query_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("scheduled_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("expected_arrival_time", pyspark.sql.types.StringType(), True)
        ])

    # 3. Operation C1: We create the DataFrame from the dataset and the schema
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ";") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)
    
    # 4. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"
    
    # 5. We add the current time stamp
    time_inputSDF = inputSDF.withColumn("my_time", F.current_timestamp())
    
    # 6. We add watermark and groupby to retrieve count
    aggSDF = time_inputSDF.withWatermark("my_time", "0 seconds").groupBy(F.window("my_time", my_frequency, my_frequency)).count()
    
    # 7. We drop the window
    SolutionSDF = aggSDF.drop("window", "window_start", "window_end")
    
    # 8. We create the DataStreamWritter, to write the results in append mode
    myDSW = SolutionSDF.writeStream\
                       .format("csv")\
                       .option("delimiter", ";")\
                       .option("path", result_dir) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .trigger(processingTime=my_frequency) \
                       .outputMode("append")
    
    # 9. We return the DataStreamWritter
    return myDSW

# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, station_number):
  
    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("station_number", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("station_name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("direction", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("day_of_week", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("query_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("scheduled_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("expected_arrival_time", pyspark.sql.types.StringType(), True)
        ])

    # 3. Operation C1: We create the DataFrame from the dataset and the schema
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ";") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)
    
    # 4. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"
    
    # 5. We add the current time stamp
    time_inputSDF = inputSDF.withColumn("my_time", F.current_timestamp())
    
    # 6. We filter records for the required station number
    FilteredSDF = time_inputSDF.filter(inputSDF['station_number'] == station_number)
    
    # 7. We project the info we are interested in 
    infoSDF = FilteredSDF.select(FilteredSDF['date'], FilteredSDF['my_time'])
    
    # 8. We retrieve unique records
    uniqueSDF = infoSDF.withWatermark("my_time", "0 seconds").dropDuplicates(["date", "my_time"])
    
    # 9. We add watermark and groupby to retrieve count
    aggSDF = uniqueSDF.withWatermark("my_time", "0 seconds").groupBy(F.window("my_time", my_frequency, my_frequency)).count().drop('window')
    
    # 10. We create the DataStreamWritter, to write the results in append mode
    myDSW = aggSDF.writeStream\
                       .format("csv")\
                       .option("delimiter", ";")\
                       .option("path", result_dir) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .trigger(processingTime=my_frequency) \
                       .outputMode("append")
    
    # 11. We return the DataStreamWriter
    return myDSW


# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, station_number):
    pass

    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("station_number", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("station_name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("direction", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("day_of_week", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("query_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("scheduled_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("expected_arrival_time", pyspark.sql.types.StringType(), True)
        ])

    # 3. Operation C1: We create the DataFrame from the dataset and the schema
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ";") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)
    
    # 4. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"
    
    # 5. We add the current time stamp
    time_inputSDF = inputSDF.withColumn("my_time", F.current_timestamp())
    
    # 6. We filter records for the required station number
    infoSDF = time_inputSDF.filter(inputSDF['station_number'] == station_number)
    
    # 7. We computer the required measurements
    StatusSDF = infoSDF.withColumn('Status', F.when(F.col('scheduled_time') >= F.col('expected_arrival_time'), "ahead").otherwise("behind"))
    
    # 8. We add watermark and retrieve the count for each measurement
    aggSDF = StatusSDF.withWatermark("my_time", "0 seconds").groupBy(F.window("my_time", my_frequency, my_frequency), StatusSDF['Status'])\
                      .count().drop("window")
   
    # 9. We create the DataStreamWritter, to write the results in append mode
    myDSW = aggSDF.writeStream\
                       .format("csv")\
                       .option("delimiter", ";")\
                       .option("path", result_dir) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .trigger(processingTime=my_frequency) \
                       .outputMode("append")
    
    # 10. We return the DataStreamWriter
    return myDSW
    
# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, station_number):
    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("station_number", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("station_name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("direction", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("day_of_week", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("query_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("scheduled_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("expected_arrival_time", pyspark.sql.types.StringType(), True)
        ])

    # 3. Operation C1: We create the DataFrame from the dataset and the schema
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ";") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)
    
    # 4. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"
    
    # 5. We add the current time stamp
    time_inputSDF = inputSDF.withColumn("my_time", F.current_timestamp())
    
    # 6. We filter records for the required station number
    FilteredSDF = time_inputSDF.filter(inputSDF['station_number'] == station_number)
    
    # 7. We add watermark and drop duplicate records on the info we are interested in
    infoSDF = FilteredSDF.select(FilteredSDF['day_of_week'], FilteredSDF['scheduled_time'], FilteredSDF['my_time'])\
                         .withWatermark("my_time", "0 seconds").dropDuplicates()
    
    # 8. We add watermark and aggregate the result 
    aggSDF = infoSDF.withWatermark("my_time", "0 seconds").groupBy(F.window("my_time", my_frequency, my_frequency), infoSDF['day_of_week'])\
                         .agg(F.collect_list(infoSDF['scheduled_time'])).drop('window')
    
    # 9. We create the DataStreamWritter, to write the results in append mode 
    myDSW = aggSDF.writeStream\
                       .format("csv")\
                       .option("delimiter", ";")\
                       .option("path", result_dir) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .trigger(processingTime=my_frequency) \
                       .outputMode("append")
    
    # 10. We return the DataStreamWriter
    return myDSW


# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, station_number, month_list):
    pass

    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("station_number", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("station_name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("direction", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("day_of_week", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("query_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("scheduled_time", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("expected_arrival_time", pyspark.sql.types.StringType(), True)
        ])

    # 3. Operation C1: We create the DataFrame from the dataset and the schema
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ";") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)
    
    # 4. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"
    
    # 5. We add the current time stamp
    time_inputSDF = inputSDF.withColumn("my_time", F.current_timestamp())
    
    # 6. We filter the records for the required station number
    FilteredSDF = time_inputSDF.filter(inputSDF['station_number'] == station_number)
    
    # 7. We project the info we are interested in
    infoSDF = FilteredSDF.select(FilteredSDF['day_of_week'],FilteredSDF['date'].substr(4, 2).alias("month"), FilteredSDF['query_time']\
                     ,FilteredSDF['expected_arrival_time'], FilteredSDF['my_time']).where(F.substring(FilteredSDF['date'],4,2).isin(month_list))
    
    # 8. We project the info we are interested in
    RenamedSDF = infoSDF.withColumn('new_date', F.concat('day_of_week', F.lit(' '), 'month')).drop('day_of_week', 'month')
    
    # 9. We find the difference in waiting times
    TimeDiffSDF = RenamedSDF.withWatermark("my_time", "0 seconds").withColumn('time1', F.to_timestamp(F.col("expected_arrival_time"),"HH:mm:ss"))\
                         .withColumn('time2', F.to_timestamp(F.col("query_time"),"HH:mm:ss"))\
                         .withColumn('DiffInSeconds', F.unix_timestamp("time1") - F.unix_timestamp("time2"))\
                         .drop('time1','time2','expected_arrival_time','query_time')
    
    # 10. We compute the average
    aggSDF = TimeDiffSDF.groupBy(F.window("my_time", my_frequency, my_frequency), TimeDiffSDF['new_date'])\
                          .avg().drop('window')
    
    # 11. We create the DataStreamWritter, to write the results in append mode
    myDSW = aggSDF.writeStream\
                       .format("csv")\
                       .option("delimiter", ";")\
                       .option("path", result_dir) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .trigger(processingTime=my_frequency) \
                       .outputMode("append")
    
    # 12. We return the DataStreamWriter
    return myDSW

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
def streaming_simulation(local_False_databricks_True, source_dir, monitoring_dir, time_step_interval, verbose):
    # 1. We get the names of the files on source_dir
    files = get_source_dir_file_names(local_False_databricks_True, source_dir, verbose)

    # 2. We get the starting time of the process
    time.sleep(time_step_interval * 0.1)

    start = time.time()

    # 2.1. If verbose mode, we inform of the starting time
    if (verbose == True):
        print("Start time = " + str(start))

    # 3. We set a counter in the amount of files being transferred
    count = 0

    # 4. We simulate the dynamic arriving of such these files from source_dir to dataset_dir
    # (i.e, the files are moved one by one for each time period, simulating their generation).
    for file in files:
        # 4.1. We copy the file from source_dir to dataset_dir#
        if local_False_databricks_True == False:
            shutil.copyfile(source_dir + file, monitoring_dir + file)
        else:
            dbutils.fs.cp(source_dir + file, monitoring_dir + file)

        # 4.2. We increase the counter, as we have transferred a new file
        count = count + 1

        # 4.3. If verbose mode, we inform from such transferrence and the current time.
        if (verbose == True):
            print("File " + str(count) + " transferred. Time since start = " + str(time.time() - start))

        # 4.3. We rename

        # 4.4. We wait the desired transfer_interval until next time slot.
        time.sleep((start + (count * time_step_interval)) - time.time())

    # 5. We wait a last time_step_interval
    time.sleep(time_step_interval)

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            result_dir,
            time_step_interval,
            verbose,
            option
           ):

    # 1. We get the DataStreamWriter object derived from the model
    dsw = None

    # Exercise 1:
    # Number of measurements per station number
    if option == 1:
        dsw = ex1(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval)

    # Exercise 2: Station 240101 (UCC WGB - Lotabeg):
    # Number of different days for which data is collected.

    # Note: Cannot be done as Structured Streaming does not support distinct operations
    # Thus, we collect: The date of each day we have collected data for.
    #                   For each of these dates, the amount of measurements we have found for it.
    # Eg: One of the entries computed by the file is:
    # 02/10/2016;5
    # Meaning that we found 5 measurement entries for day 02/10/2016 and station 240101.
    if option == 2:
        dsw = ex2(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, 240101)

    # Exercise 3: Station 240561 (UCC WGB - Curraheen):
    # Number of buses arriving ahead and behind schedule.
    if option == 3:
        dsw = ex3(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, 240561)

    # Exercise 4: Station 241111 (CIT Technology Park - Lotabeg):
    # List of buses scheduled per day of the week.

    # Note: Cannot be done as Structured Streaming does not support distinct operations
    # Thus, we collect: The date of each day we have collected data for.
    #                   For each of these dates, the amount of measurements we have found for it.
    # Eg: One of the entries computed by the file is:
    # Tuesday;16:11:00;96
    # Meaning that we found 96 measurement entries for Tuesday and bus_scheduled at 16:11:00 at station 241111
    if option == 4:
        dsw = ex4(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, 241111)

    # Exercise 5: Station 240491 (Patrick Street - Curraheen):
    # Average waiting time per day of the week during the Semester1 months. Sort the entries by decreasing average waiting time.

    # Note: Cannot sort as sorting is only supported after an aggregation and Complete mode (not append mode).
    if option == 5:
        dsw = ex5(spark, monitoring_dir, checkpoint_dir, result_dir, time_step_interval, 240491, ['09','10','11'])

    # 2. We get the StreamingQuery object derived from starting the DataStreamWriter
    ssq = dsw.start()

    # 3. We stop the StreamingQuery to finish the application
    ssq.awaitTermination(time_step_interval)

    # 4. We simulate the streaming arrival of files (i.e., one by one) from source_dir to monitoring_dir
    streaming_simulation(local_False_databricks_True, source_dir, monitoring_dir, time_step_interval, verbose)

    # 5. Once we have transferred all files and processed them, we are done.
    # Thus, we stop the StreamingQuery
    ssq.stop()


# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    option = 1

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = "/home/nacho/CIT/Tools/MyCode/Spark/"
    my_databricks_path = "/"

    source_dir = "Filestore/tables/7_Assignments/A02/my_dataset_complete/"
    monitoring_dir = "Filestore/tables/7_Assignments/A02/my_monitoring/"
    checkpoint_dir = "Filestore/tables/7_Assignments/A02/my_checkpoint/"
    result_dir = "Filestore/tables/7_Assignments/A02/my_result/"

    if local_False_databricks_True == False:
        source_dir = my_local_path + source_dir
        monitoring_dir = my_local_path + monitoring_dir
        checkpoint_dir = my_local_path + checkpoint_dir
        result_dir = my_local_path + result_dir
    else:
        source_dir = my_databricks_path + source_dir
        monitoring_dir = my_databricks_path + monitoring_dir
        checkpoint_dir = my_databricks_path + checkpoint_dir
        result_dir = my_databricks_path + result_dir

    # 4. We set the Spark Streaming parameters

    # 4.1. We specify the time interval each of our micro-batches (files) appear for its processing.
    time_step_interval = 20

    # 4.2. We configure verbosity during the program run
    verbose = False

    # 5. We remove the directories
    if local_False_databricks_True == False:
        # 5.1. We remove the monitoring_dir
        if os.path.exists(monitoring_dir):
            shutil.rmtree(monitoring_dir)

        # 5.2. We remove the result_dir
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)

        # 5.3. We remove the checkpoint_dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    else:
        # 5.1. We remove the monitoring_dir
        dbutils.fs.rm(monitoring_dir, True)

        # 5.2. We remove the result_dir
        dbutils.fs.rm(result_dir, True)

        # 5.3. We remove the checkpoint_dir
        dbutils.fs.rm(checkpoint_dir, True)

    # 6. We re-create the directories again
    if local_False_databricks_True == False:
        # 6.1. We re-create the monitoring_dir
        os.mkdir(monitoring_dir)

        # 6.2. We re-create the result_dir
        os.mkdir(result_dir)

        # 6.3. We re-create the checkpoint_dir
        os.mkdir(checkpoint_dir)
    else:
        # 6.1. We re-create the monitoring_dir
        dbutils.fs.mkdirs(monitoring_dir)

        # 6.2. We re-create the result_dir
        dbutils.fs.mkdirs(result_dir)

        # 6.3. We re-create the checkpoint_dir
        dbutils.fs.mkdirs(checkpoint_dir)

    # 7. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 8. We call to our main function
    my_main(spark,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            result_dir,
            time_step_interval,
            verbose,
            option
           )