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
from pyspark.sql.types import LongType

# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(spark, my_dataset_dir):
    # 1. We define the Schema of our DF.
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

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
                        .option("delimiter", ";") \
                        .option("quote", "") \
                        .option("header", "false") \
                        .schema(my_schema) \
                        .load(my_dataset_dir)

    # 3. We count the number of entries in inputDF
    resval = inputDF.count()
    
    # 4. We print the result
    print(resval)

# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(spark, my_dataset_dir, station_number):
    # 1. We define the Schema of our DF.
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

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
  
    # 3. We filter records for the required station number
    infoDF = inputDF.filter(inputDF['station_number'] == station_number)
    
    # 4. We retrieve the unique calender days and find the count
    rowcount = infoDF.dropDuplicates(['date']).count()
    
    # 5. We print the result
    print(rowcount)


# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(spark, my_dataset_dir, station_number):
    # 1. We define the Schema of our DF.
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

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
    
    # 3. We filter records for the required station number
    infoDF = inputDF.filter(inputDF['station_number'] == station_number)
    
    # 4. We persist the data
    infoDF.persist()
    
    # 5. We retrieve the count for 1st computation
    Accum1 = infoDF.select().where(infoDF['scheduled_time'] >= infoDF['expected_arrival_time']).count()
    
    # 6. We retrieve the count for 2nd computation
    Accum2 = infoDF.select().where(infoDF['scheduled_time'] < infoDF['expected_arrival_time']).count()
    
    # 7. We print the result
    print((Accum1, Accum2))
    
# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(spark, my_dataset_dir, station_number):
    # 1. We define the Schema of our DF.
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

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)

    # 3. We filter records for the required station number
    FilteredDF = inputDF.filter(inputDF['station_number'] == station_number)
    
    # 4. We select the unique records 
    infoDF = FilteredDF.select(FilteredDF['day_of_week'], FilteredDF['scheduled_time']).dropDuplicates()
    
    # 5. We sort the records
    SortedDF = infoDF.orderBy(infoDF['scheduled_time'])
    
    # 6. We aggreate to a list and collect the result
    resVal = SortedDF.groupBy('day_of_week').agg(F.collect_list(SortedDF['scheduled_time']).alias('bus_scheduled_times')).collect()
    
    # 7. We print the result
    for item in resVal:
      print(item)
    
# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(spark, my_dataset_dir, station_number, month_list):
    # 1. We define the Schema of our DF.
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

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
    
    # 3. We filter records for the required station number
    FilteredDF = inputDF.filter(inputDF['station_number'] == station_number)
   
    # 4. We project the information we are intersted into
    infoDF = FilteredDF.select(FilteredDF['day_of_week'],FilteredDF['date'].substr(4, 2).alias("month"), FilteredDF['query_time']\
                     ,FilteredDF['expected_arrival_time']).where(F.substring(FilteredDF['date'],4,2).isin(month_list))
    
    # 5. We produce the format required for the result
    RenamedDF = infoDF.withColumn('new_date', F.concat('day_of_week', F.lit(' '), 'month')).drop('day_of_week', 'month')
      
    # 6. We calculate the waiting time difference  
    TimeDiffDF = RenamedDF.withColumn('time1', F.to_timestamp(F.col("expected_arrival_time"),"HH:mm:ss"))\
                         .withColumn('time2', F.to_timestamp(F.col("query_time"),"HH:mm:ss"))\
                         .withColumn('DiffInSeconds', F.unix_timestamp("time1") - F.unix_timestamp("time2"))\
                         .drop('time1','time2','expected_arrival_time','query_time')
    
    # 7. We compute the average
    AverageDF = TimeDiffDF.groupBy('new_date').avg().withColumnRenamed('avg(DiffInSeconds)','Average_time')
    
    # 8. We collect the results
    resVal = AverageDF.orderBy('Average_time').collect()
    
    # 9. We print the results
    for item in resVal:
      print(item)
    
    
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, option):
    # Exercise 1:
    # Number of measurements per station number
    if option == 1:
        ex1(spark, my_dataset_dir)

    # Exercise 2: Station 240101 (UCC WGB - Lotabeg):
    # Number of different days for which data is collected.
    if option == 2:
        ex2(spark, my_dataset_dir, 240101)

    # Exercise 3: Station 240561 (UCC WGB - Curraheen):
    # Number of buses arriving ahead and behind schedule.
    if option == 3:
        ex3(spark, my_dataset_dir, 240561)

    # Exercise 4: Station 241111 (CIT Technology Park - Lotabeg):
    # List of buses scheduled per day of the week.
    if option == 4:
        ex4(spark, my_dataset_dir, 241111)

    # Exercise 5: Station 240491 (Patrick Street - Curraheen):
    # Average waiting time per day of the week during the Semester1 months. Sort the entries by decreasing average waiting time.
    if option == 5:
        ex5(spark, my_dataset_dir, 240491, ['09','10','11'])


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

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, option)