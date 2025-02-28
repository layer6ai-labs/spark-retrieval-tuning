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
export SPARK_HOME="/usr/lib/spark"

export TPCDS_INPUT_DATA_DIR="hdfs:///data/$size/"
export TPCDS_QUERY=$query


cd "/home/hadoop/tpcds-spark/data_tpcds_${size}_${query}_${id}"

echo "++++++++++++++++++++++++++++"
echo "Running TPC-DS query $query for dataset size $size"
echo "Data location: $TPCDS_INPUT_DATA_DIR"
echo "Query: $query"


/home/hadoop/tpcds-spark/ubin/run-tpcds.sh --data-location ${TPCDS_INPUT_DATA_DIR} --query-filter ${query} \
               --conf spark.sql.queryExecutionListeners=$SPARK_QUERY_LISTENERS\
               --jars $SPARK_LISTENERS_JAR \
               --conf spark.extraListeners=$SPARK_EXTRA_LISTENERS \
                ${additional_confs} \
               --conf spark.listeners.layer6.appName=tpcds_${size}_${query} \
               --conf spark.listeners.layer6.appId=${id} \
               


hdfs dfs -rmr hdfs:///output_result