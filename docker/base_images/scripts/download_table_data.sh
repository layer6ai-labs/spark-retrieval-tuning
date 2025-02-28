#!/bin/bash

BASE_PATH=/home/data_dir
MOUNT_BASE_PATH=/mnt/spark_tuning

download_tpch_data() {
    local size=$1
    echo "Writing tpch data of size $size from remote into hdfs"
    hadoop distcp s3://layer6/sparktune/data/tpch/$size/ hdfs:///data/$size
}

# Function to perform actions for tpcds
download_tpcds_data() {
    local size=$1
    echo "Writing tpcds data of size $size from remote into hdfs"
    hadoop distcp s3://layer6/sparktune/data/tpcds/$size/ hdfs:///data/$size
    echo "File uploaded to hdfs"
}


# Main script
data_password=$1
dataset=$2
size=$3
job_type=$4 

case "$dataset" in
    "tpch")
        download_tpch_data "$size"
        ;;
    "tpcds")
        download_tpcds_data "$size"
        ;;
    *)
        echo "Unknown output"
        ;;
esac
