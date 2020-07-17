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
        res = (int(params[0]),
               str(params[1]),
               float(params[2]),
               float(params[3]),
               str(params[4]),
               int(params[5]),
               int(params[6])
               )

    # 5. We return res
    return res
  
def my_functionex5(list1):
  
    biketaken = 0
    bikereturned = 0
    previous_val = list1[0]
    
    for bikes in list1[1:]:
      diff = (previous_val - bikes)
      if  diff >= 0:
        biketaken += abs(diff)
      else:
        bikereturned += abs(diff)
      previous_val = bikes
      
    return (biketaken,bikereturned)
    
def my_functionex3(list1):
  
    result = []
    previous_val = 1
    
    for (bikes,time) in list1:
      if bikes == 0 and previous_val > 0:
        result.append(time)
      previous_val = bikes
    if len(result) > 0:
      return result

# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(sc, my_dataset_dir):
  
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
    
    # 3. Filter the stations with inactive status and records having non zero bikes available
    filterinactivelinesRDD = allLinesRDD.filter(lambda x : x[0] == 0 and x[5] == 0)
    
    # 4. We project just the info we are interested into
    StationbikespairRDD = filterinactivelinesRDD.map(lambda x: (x[1],x[5]))
    
    # 5. For each Station, accumulate the row values and the number of rows
    combineRDD = StationbikespairRDD.combineByKey(lambda val: (val, 1),
                                        lambda accum, new_val: (accum[0] + new_val, accum[1] + 1),
                                        lambda final_accum1, final_accum2: (final_accum1[0] + final_accum2[0], final_accum1[1] + final_accum2[1])
                                       )
    
    # 6. We project just the info we are interested into
    BikerunoutRDD = combineRDD.mapValues(lambda x : x[1])
    
    # 7. Sort by the bikes run out count in descending order 
    RunoutSortedRDD = BikerunoutRDD.sortBy(lambda x: x[1],False)
    
    # 8. Count the number of resulting rows
    Rowcount = RunoutSortedRDD.count()
    
    # 9. Collect the resulting rows 
    resVal = RunoutSortedRDD.collect()

    # 10. Print Rowcount
    print(Rowcount)
    
    # 11. Print each record in result
    for item in resVal:
      print(item)

    
# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(sc, my_dataset_dir):
    
    # 1. We load the file for the day Sunday 28th August 2017 into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir+"bikeMon_20170827.csv/")
    
    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
    
    # 3. Filter the stations with inactive status
    filteractivelinesRDD = allLinesRDD.filter(lambda x : x[0] == 0)
    
    # 4. We project just the info we are interested into
    StationbikespairRDD = filteractivelinesRDD.map(lambda x: ((x[1]+' '+x[4][11:13]), x[5]))
    
    # 5. For each Station and hour group, accumulate the vslue(Bikes available) and the number of rows
    AccumRDD = StationbikespairRDD.combineByKey(lambda val: (val, 1),
                                                lambda accum, new_val: (accum[0] + new_val, accum[1] + 1),
                                                lambda final_accum1, final_accum2: (final_accum1[0] + final_accum2[0], final_accum1[1] + final_accum2[1])
                                                )
    
    # 6. Calculate the average amount of bikes
    AverageRDD = AccumRDD.mapValues(lambda x: x[0] / x[1])
    
    # 7. Sort the rsulting rows by Station and hour group
    resval = AverageRDD.sortByKey().collect()
    
    # 8. Print each record in result
    for item in resval:
      print(item)

# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(sc, my_dataset_dir):
    
    # 1. We load the file for the day Sunday 28th August 2017 into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir+"bikeMon_20170827.csv/")
   
    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
    
    # 3. Filter the stations with inactive status
    filteractivelinesRDD = allLinesRDD.filter(lambda x : x[0] == 0)
    
    # 4. We project just the info we are interested into
    StationHourbikesRDD = filteractivelinesRDD.map(lambda x: (x[1], (x[5],x[4][11:16])))
    
    # 5. For each Station accumulate the value(Bikes available) as a list and the number of rows
    combineRDD = StationHourbikesRDD.combineByKey(lambda x:[x],lambda x,y:x+[y],lambda x,y:x+y)
    
    # 6. For each list of values, apply the user defined function to determine the number of first time runout in each set runout instances
    usefunctionRDD = combineRDD.mapValues(lambda x : my_functionex3(x)).filter(lambda x: x[1] != None)
    
    # 7. Arrange the rows in required format and Sort the resulting rows by run out time and Station name
    sortfinalRDD = usefunctionRDD.flatMap(lambda y: [(val, y[0]) for val in y[1]]).sortBy(lambda x: (x[0],x[1]))
    
    # 8. Collect the resulting rows
    resval = sortfinalRDD.collect()
    
    # 9. Print each record in result
    for item in resval:
      print(item)

# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(sc, my_dataset_dir, ran_out_times):
    
    # 1. We load the file for the day Sunday 28th August 2017 into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir+"bikeMon_20170827.csv/")
    
    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
    
    # 3. Filter the stations with inactive status
    filteractivelinesRDD = allLinesRDD.filter(lambda x : x[0] == 0 and x[4][11:19] in ran_out_times)
    
    # 4. We project just the info we are interested into
    HourStationbikesRDD = filteractivelinesRDD.map(lambda x: (x[4][11:19], (x[1],x[5])))
    
    # 5. For each run out time, retrieve the station with maximum number of bikes available
    maxperkeyRDD = HourStationbikesRDD.reduceByKey(lambda x1, x2 : x1 if x1[1] > x2[1] else x2).sortByKey()
    
    # 6. Collect the resulting rows
    resval = maxperkeyRDD.collect()

    # 7. Print each record in result 
    for item in resval:
      print(item)
      
# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(sc, my_dataset_dir):
   
    # 1. We load the dataset into an inputRDD
    inputRDD = sc.textFile(my_dataset_dir)
    
    # 2. We process each line to get the relevant info
    allLinesRDD = inputRDD.map(process_line)
    
    # 3. Filter the stations with inactive status and the unwanted runout times
    filteractivelinesRDD = allLinesRDD.filter(lambda x : x[0] == 0)
    
    # 4. We project just the info we are interested into
    StationbikesRDD = filteractivelinesRDD.map(lambda x: ((x[1],x[4][0:10]),x[5]))
    
    # 5. For each Station, accumulate the values(Bikes available) as a list
    combineRDD = StationbikesRDD.combineByKey(lambda x:[x],lambda x,y:x+[y],lambda x,y:x+y)
    
    # 6. From the above list of values, for each station, Calculate the number of bikes Taken and returned per day with a user defined function
    usefunctionRDD = combineRDD.mapValues(lambda x : my_functionex5(x))
    
    # 7. We project just the info we are interested into
    removeDayKeyRDD = usefunctionRDD.map(lambda x : (x[0][0],x[1]))
    
    # 8. Accumulate the bikes Taken and returned for each Station on all days
    reduceRDD = removeDayKeyRDD.reduceByKey(lambda x, y : (x[0]+y[0],x[1]+y[1]))
    
    # 9. Sort the resulting rows by the sum of bikes taken + bikes given back in descending order
    sortedRDD = reduceRDD.sortBy(lambda x: (x[1][0] + x[1][1]),False)
    
    # 10. Collect the resulting rows
    resval = sortedRDD.collect()

    # 11. Print each record in result 
    for item in resval:
      print(item)
    

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, option, ran_out_times):
    # Exercise 1: Number of times each station ran out of bikes (sorted decreasingly by station).
    if option == 1:
        ex1(sc, my_dataset_dir)

    # Exercise 2: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Average amount of bikes per station and hour window (e.g. [9am, 10am), [10am, 11am), etc. )
    if option == 2:
        ex2(sc, my_dataset_dir)

    # Exercise 3: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Get the different ran-outs to attend.
    #             Note: n consecutive measurements of a station being ran-out of bikes has to be considered a single ran-out,
    #                   that should have been attended when the ran-out happened in the first time.
    if option == 3:
        ex3(sc, my_dataset_dir)

    # Exercise 4: Pick one busy day with plenty of ran outs -> Sunday 28th August 2017
    #             Get the station with biggest number of bikes for each ran-out to be attended.
    if option == 4:
        ex4(sc, my_dataset_dir, ran_out_times)

    # Exercise 5: Total number of bikes that are taken and given back per station (sorted decreasingly by the amount of bikes).
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

    ran_out_times = ['06:03:00', '06:03:00', '08:58:00', '09:28:00', '10:58:00', '12:18:00',
                     '12:43:00', '12:43:00', '13:03:00', '13:53:00', '14:28:00', '14:28:00',
                     '15:48:00', '16:23:00', '16:33:00', '16:38:00', '17:09:00', '17:29:00',
                     '18:24:00', '19:34:00', '20:04:00', '20:14:00', '20:24:00', '20:49:00',
                     '20:59:00', '22:19:00', '22:59:00', '23:14:00', '23:44:00']

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
    my_main(sc, my_dataset_dir, option, ran_out_times)
