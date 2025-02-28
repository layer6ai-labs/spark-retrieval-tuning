#!/bin/bash

set -e

id=$1
size=$2
query=$3
shift 3
additional_confs="$@"

export SPARK_LISTENERS_JAR="/home/hadoop/spark-listeners/target/scala-2.12/layer6-spark-listeners-assembly-1.0.jar"
export SPARK_EXTRA_LISTENERS="ai.layer6.spark.listeners.TaskMetricsListener,ai.layer6.spark.listeners.StageMetricsListener"
export SPARK_QUERY_LISTENERS="ai.layer6.spark.listeners.QueryPlanListener"
export SPARK_WORKER_OPTS="$SPARK_WORKER_OPTS -Dspark.worker.cleanup.enabled=true -Dspark.worker.cleanup.appDataTtl=1800"

export TPCH_INPUT_DATA_DIR="hdfs:///data/$size"
export TPCH_QUERY_OUTPUT_DIR="hdfs:///results"
export TPCH_QUERY=$query

cd "/home/hadoop/data_tpch_${size}_${query}_${id}"
            
spark-submit \
 --jars $SPARK_LISTENERS_JAR \
 --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS \
 --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS \
 --conf spark.listeners.layer6.appName=tpch_${size}_${query} \
 --conf spark.listeners.layer6.appId=$id \
 ${additional_confs} \
 --class "main.scala.TpchQuery" /home/hadoop/tpch-spark/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar $TPCH_QUERY

hdfs dfs -rmr hdfs:///results