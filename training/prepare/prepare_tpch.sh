#!/bin/bash

set -e

./../docker/build.sh

data_password=$1
dataset_size=$2

cd ../docker/tpch-spark3.5
docker-compose up --detach
sleep 30 # for namenode to finish setting up

docker exec -it tpch-spark35_tpch_1 /scripts/download_table_data.sh $data_password tpch $dataset_size


