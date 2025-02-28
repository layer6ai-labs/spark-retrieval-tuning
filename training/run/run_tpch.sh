set -e

./../docker/build.sh

data_password=$1
dataset_size=$2
run_id=$3
queries=$4

./prepare/prepare_tpch.sh $data_password $dataset_size
python ./src/optimize_queries_training.py --size $dataset_size --id $run_id --dataset tpch --queries $4
./teardown/teardown_tpch.sh