#!/bin/bash

set -e

data_password=$1
dataset_size=$2

# Verify that the parameters were passed in
if [[ -z $data_password || -z $dataset_size ]]; then
    echo "Error: Missing parameters. Please provide data password and dataset size."
    exit 1
fi


./../docker/build.sh


cd ../docker/tpcds-spark3.5
docker-compose up --detach
echo "Waiting for namenode to finish setting up"
sleep 60 # for namenode to finish setting up

echo "Downloading data for TPC-DS"
docker exec -it tpcds-spark35_tpcds_1 /scripts/download_table_data.sh $data_password tpcds $dataset_size

