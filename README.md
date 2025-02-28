<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Zero-Execution Retrieval-Augmented Configuration Tuning of Spark Applications 

This repo experiments on various spark config tuning methods.

To start up a spark cluster, run:
```
cd docker/spark
docker-compose up
```

This will create a spark cluster with 2 workers with the master URL of `spark://localhost:7077`

If you have pyspark setup, you can run `pyspark --master spark://localhost:7077` and start running spark queries on this cluster.docker exec -it tpch-spark35_tpch_1 bash

## Build base images:
Run the following command to build all base images:
```
./docker/build.sh
```


To build specific base images, pass one of the following as a positional argument: `spark3, spark3.5`
```
./docker/build.sh spark3
```
## Mount input data drive
If disk usage space is a concern on the current machine, we sugggest moving the data to a separate machine and mounting that machine. 
To avoid filling up disk space when running benchmarks on a local machine, we mount this remote drive to the local machine directly.  Run the script `sudo scripts/mount_storage.sh` to mount the drive to `/mnt/spark_tuning` directory.  Run the script `sudo scripts/unmount_storage.sh` to clean up the mount. 


## TPC-H
Go inside the tpc-h container:
```
docker exec -it tpch-spark35_tpch_1 bash
```

Inside the container, run the tpc-h benchmark:
```
./scripts/download_table_data.sh <password to download data> tpch <size {1g|10g|30g|100g|300g}>
export TPCH_INPUT_DATA_DIR="hdfs://spark-master:9000/data/<size>"
export TPCH_QUERY_OUTPUT_DIR="hdfs://spark-master:9000/results"
cd /tpch-spark
/opt/spark/bin/spark-submit\
    --master spark://spark-master:7077 --jars $SPARK_LISTENERS_JAR --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS --class "main.scala.TpchQuery" target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar
```

## TPC-DS

Go inside the tpc-ds container: 
```
docker exec -it tpcds-spark35-tpcds-1 bash
```


### Generate the data
```
cd ubin
```
There are multiple generation scripts, but I've found the third one works best. 
```
./dsdgen3.sh --output-location ../data --scale-factor 1
```
Create HDFS directory and copy over the data
```
hdfs dfs -mkdir hdfs://spark-master:9000/tpcds_data
hdfs dfs -copyFromLocal ../data/* hdfs://spark-master:9000/tpcds_data
```

### Run the Queries

All Queries
```
./run-tpcds.sh --data-location hdfs://spark-master:9000/tpcds_data \
               --master spark://spark-master:7077 \
               --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS\
               --jars $SPARK_LISTENERS_JAR \
               --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS  

```

Subset
```
./run-tpcds.sh --data-location hdfs://spark-master:9000/tpcds_data --query-filter "q2,q10" \
               --master spark://spark-master:7077 \
               --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS\
               --jars $SPARK_LISTENERS_JAR \
               --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS  
```

 Range
```
 ./run-tpcds.sh --data-location hdfs://spark-master:9000/tpcds_data --query-filter "q11-27" 
               --master spark://spark-master:7077 \
               --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS\
               --jars $SPARK_LISTENERS_JAR \
               --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS  
```
You can view the results in `tpcds-spark/output_result` directory. 

### EMR
1. spin up 5 * r6g.2xlarge   1 master and 4 core nodes
2. disable dynamic allocation `spark.dynamicAllocation.enabled`
3. `ssh -i ~/emr.pem hadoop@<ip>`
4. Clone listener repo, tpch repo and tpcds repo along side tuning repo
5. Run `emr_setup`.sh
6. Run `(cd ~/tpch-spark && exec ~/sbt/bin/sbt package)` or `(cd ~/tpcds-spark && exec ~/sbt/bin/sbt package)`
7. ./scripts/download_table_data.sh spark2023 tpch 10g
8. go into `spark-tuning` and do `pip install -r requirements.txt`
9. go into the training directory and run the scripts you need to

# Data
The raw data containing 19360 spark application executions with varying configuration parameters is publicly available at `s3://l6lab/sparktune/raw`. In it, you will find `data_<tpch|tpcds>_<100g|250g|500g|750g>_<query>_emr_<trial_number>` 
which contains the execution metrics, logical plans, and execution runtime of that specific run. There is also `<tpch|tpcds>_<100g|250g|500g|750g>_<query>_emr.db`
which contains optuna trials data, indicating the spark configuration parameters that the trial was executed with.

# License
This data and code is licensed under the MIT License, copyright by Layer 6 AI.
