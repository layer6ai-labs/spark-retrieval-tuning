# Spark Listeners

Collection of useful spark listeners

To build the package:
`sbt assembly` 

## Stage and Task Listeners

To save the stage and task metrics raw output metrics to a JSON file, the application name and application ID are used to denote the file. If the same application is run multiple times with the same name/ID, the results are appended to the existing JSON file.

On the other hand, the aggregated examples are saved to a CSV file. The file is also denoted by the application name and application ID. Similar to the JSON file, if the same application is run multiple times with the same name/ID, the results are appended to the existing CSV file.


### Running the Stage and Task Listeners

The easiest way to run it is by passing the listeners as command line arguments to the spark job. 

```
spark-shell --conf spark.logConf=true --conf spark.extraListeners=ai.layer6.spark.listeners.StageMetricsListener,ai.layer6.spark.listeners.TaskMetricsListener --conf spark.listeners.layer6.appName=my_program --conf spark.listeners.layer6.appId=1 --jars target/scala-2.12/layer6-spark-listeners-assembly-1.0.jar 
```

# Query Plan Listener

This listener is used to save the query execution plans to a JSON file. This includes the logical, physical, optimized and durations of the plans for each query in a given application. 

The results in the JSON file are separated by application and each application can have multiple queries. If there are multiple queries in a given application, then they are appended to a list of queries for that application. If the application is run more than once, then the application queries for the second run is appended to the JSON file. So the JSON file is a list of application run(s) and their query plan(s).

### Running the Query Plan Listener

```
import ai.layer6.spark.listeners._
spark.listenerManager.register(new QueryPlanListener())
```
