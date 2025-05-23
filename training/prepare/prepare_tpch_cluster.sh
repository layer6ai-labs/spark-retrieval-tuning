#!/bin/bash

set -e

./../docker/build.sh

data_password=$1
dataset_size=$2

cd ../docker/tpch-spark3.5
docker stack deploy tpch-dist -c compose_cluster.yaml
sleep 30 # for namenode to finish setting up

docker exec -it $(docker ps -q -f name="tpch-dist_tpch") /scripts/download_table_data.sh $data_password tpch $dataset_size


