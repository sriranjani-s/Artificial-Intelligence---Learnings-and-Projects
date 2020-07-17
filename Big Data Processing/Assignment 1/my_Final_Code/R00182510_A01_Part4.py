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
import pyspark.sql.functions as F
import pyspark.sql.types

# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(spark, my_dataset_dir):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("status", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("dateStatus", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("bikesAvailable", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("docksAvailable", pyspark.sql.types.IntegerType(), True)
         ])

    # 2. Operation C2: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
    
    # 3. Operation T1: Filter inactive bike stations and project just the info we are interested in
    infoDF = inputDF.select(inputDF['name'],inputDF['bikesAvailable']).where((inputDF['bikesAvailable'] == 0) & (inputDF['status'] == 0))
    
    # 4. Operation T2: Groupby station name and rerieve the number of rows in each group using count
    RunoutsDF = infoDF.groupBy(["name"]).count()
    
    # 5. Operation T3: Sort by the count values in descending order
    resultDF = RunoutsDF.orderBy(RunoutsDF['count'].desc())
    
    # 6. Operation T4: Persist the DF
    resultDF.persist()
    
    # 7. Operation T5: Get count of total number of rows in the resulting Dataframe
    Rowcount = resultDF.count()
    
    # 8. Operation A1: Collect the rows
    resval = resultDF.collect()
    
    # 9. Print the Result
    print(Rowcount)
    for item in resval:
      print(item)

# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(spark, my_dataset_dir):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("status", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("dateStatus", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("bikesAvailable", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("docksAvailable", pyspark.sql.types.IntegerType(), True)
         ])

    # 2. Operation C2: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir+"bikeMon_20170827.csv/")
    
    # 3. Operation T1: Filter inactive bike stations and project just the info we are interested in
    infoDF = inputDF.select(inputDF['name'],inputDF['dateStatus'].substr(11, 3).alias("hour"), inputDF['bikesAvailable']).where(inputDF['status'] == 0)
    
    # 4. Operation T2: Group by Station name and hour and calculate the average bikes available for each group
    BikesperhourDF = infoDF.groupBy(["name", "hour"]).avg("bikesAvailable")    
    
    # 5. Operation A1: Sort by the station name and hour and collect the rows
    resval = BikesperhourDF.orderBy(BikesperhourDF["name"], BikesperhourDF["hour"]).collect()
    
    # 6. Print the result
    for item in resval:
      print(item)    

# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(spark, my_dataset_dir):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("status", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("dateStatus", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("bikesAvailable", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("docksAvailable", pyspark.sql.types.IntegerType(), True)
         ])

    # 2. Operation C2: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir+"bikeMon_20170827.csv/")
  
    # 3. Operation T1: Filter inactive bike stations
    filteredDF = inputDF.filter(inputDF['status'] == 0)
    
    # 4. Operation T2:  We project just the info we are interested in
    infoDF = filteredDF.select(filteredDF['name'],filteredDF['dateStatus'].substr(12, 5).alias("time"), filteredDF['bikesAvailable'])
    
    # 5. Operation T3: We create the window we want to perform an operation on
    my_window = pyspark.sql.Window().partitionBy().orderBy(inputDF['name'])
    
    # 6. Operation T4: We apply the window based transformation to create a new column containing previous value of bikes available
    includePreviousDF = infoDF.withColumn("prev_val", F.lag(inputDF['bikesAvailable'], 1).over(my_window))
    
    # 7. Operation T5: We replace the Null value generated in the first row with '1' 
    replaceNullDF = includePreviousDF.withColumn("prev_val", F.coalesce(includePreviousDF['prev_val'] , F.lit(1))) 
    
    # 8. Operation T6: Select the rows with first time bike runouts from all the runout instances 
    runoutsDF = replaceNullDF.select(replaceNullDF['name'],replaceNullDF['time']).where((replaceNullDF['bikesAvailable'] == 0) & (replaceNullDF['prev_val'] > 0))
    
    # 9. Opeartion A1: Sort by the runout time and collect the rows
    resval = runoutsDF.orderBy(runoutsDF['time']).collect()
    
    # 10. Print the result
    for item in resval:
      print(item)

# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(spark, my_dataset_dir, ran_out_times):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("status", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("dateStatus", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("bikesAvailable", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("docksAvailable", pyspark.sql.types.IntegerType(), True)
         ])

    # 2. Operation C2: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir+"bikeMon_20170827.csv/")
    
    # 3. Operation T1: Filter inactive bike stations and project just the info we are interested in
    infoDF = inputDF.select(inputDF['name'], inputDF['dateStatus'], inputDF['bikesAvailable']).where(inputDF['status'] == 0)
    
    # 4. Operation T2: Filter only the rows that belong to the run out time specified in the given input
    runoutsDF = infoDF.filter(F.substring(inputDF['dateStatus'],12,8).isin(ran_out_times))

    # 5. Operation T3: Sort by bikes available in descending order to retrieve the station with maximum available bikes
    MaxavailabilityDF = runoutsDF.orderBy(runoutsDF['bikesAvailable'].desc()).dropDuplicates(['dateStatus'])
    
    # 6. Operation A1: Sort by the run out date and time and collect the rows
    resval = MaxavailabilityDF.orderBy(MaxavailabilityDF['dateStatus']).collect()
    
    # 7. Print the result
    for item in resval:
      print(item)

# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(spark, my_dataset_dir):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("status", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("name", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), True),
         pyspark.sql.types.StructField("dateStatus", pyspark.sql.types.StringType(), True),
         pyspark.sql.types.StructField("bikesAvailable", pyspark.sql.types.IntegerType(), True),
         pyspark.sql.types.StructField("docksAvailable", pyspark.sql.types.IntegerType(), True)
         ])

    # 2. Operation C1: We create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ";") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
    
    # 3. Operation T1: Filter inactive bike stations and project just the info we are interested in
    infoDF = inputDF.select(inputDF['name'], inputDF['dateStatus'], inputDF['bikesAvailable']).where(inputDF['status'] == 0)
    
    # 4. Operation T2: We create the window we want to perform an operation on
    my_window = pyspark.sql.Window().partitionBy(infoDF['name'],F.substring(infoDF['dateStatus'],0,10)).orderBy(infoDF['dateStatus'])
    
    # 5. Operation T3: We apply the window based transformation to create a new column containing the difference in bikes avaiable between the current and previous row
    bikeDiffDF = infoDF.withColumn('diff',infoDF['bikesAvailable'] - F.lag(infoDF['bikesAvailable'] , 1).over(my_window))
    
    # 6. Operation T4: Persist the current Dataframe
    bikeDiffDF.persist()
    
    # 7. Operation T5: Calculate the bikes taken aggregate for every station
    bikestakenDF = bikeDiffDF.where(bikeDiffDF['diff'] < 0).groupBy(bikeDiffDF['name']).agg(F.sum(F.abs(bikeDiffDF['diff'])).alias('bikesTaken'))                              

    # 8. Operation T5: Calculate the bikes given back aggregate for every station
    bikesGivenBackDF = bikeDiffDF.where(bikeDiffDF['diff'] >= 0).groupBy(bikeDiffDF['name']).agg(F.sum(F.abs(bikeDiffDF['diff'])).alias('bikesGivenBack'))

    # 9. Operation T6: Join the bikes taken and bikes given back dataframes
    resultDF = bikestakenDF.join(bikesGivenBackDF, 'name')                                                                                                                       
    
    # 10. Operation A1: For every station, Sort by the sum of bikes taken and given back and collect the rows
    resval = resultDF.orderBy((resultDF['bikesTaken'] + resultDF['bikesGivenBack']).desc()).collect()
    
    # 11. Print the result
    for item in resval:
      print(item)
    

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, option, ran_out_times):
    # Exercise 1: Number of times each station ran out of bikes (sorted decreasingly by station).
    if option == 1:
        ex1(spark, my_dataset_dir)

  # Exercise 2: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Average amount of bikes per station and hour window (e.g. [9am, 10am), [10am, 11am), etc. )
    if option == 2:
        ex2(spark, my_dataset_dir)

    # Exercise 3: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Get the different ran-outs to attend.
    #             Note: n consecutive measurements of a station being ran-out of bikes has to be considered a single ran-out,
    #                   that should have been attended when the ran-out happened in the first time.
    if option == 3:
        ex3(spark, my_dataset_dir)

    # Exercise 4: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Get the station with biggest number of bikes for each ran-out to be attended.
    if option == 4:
        ex4(spark, my_dataset_dir, ran_out_times)

    # Exercise 5: Total number of bikes that are taken and given back per station (sorted decreasingly by the amount of bikes).
    if option == 5:
        ex5(spark, my_dataset_dir)

# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    option = 3

    ran_out_times = ["06:03:00", "06:03:00", "08:58:00", "09:28:00", "10:58:00", "12:18:00",
                     '12:43:00', '12:43:00', '13:03:00', '13:53:00', '14:28:00', '14:28:00',
                     '15:48:00', '16:23:00', '16:33:00', '16:38:00', '17:09:00', '17:29:00',
                     '18:24:00', '19:34:00', '20:04:00', '20:14:00', '20:24:00', '20:49:00',
                     '20:59:00', '22:19:00', '22:59:00', '23:14:00', '23:44:00'
                     ]

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

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, option, ran_out_times)
