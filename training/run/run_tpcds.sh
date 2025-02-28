set -e

./../docker/build.sh

data_password=$1
dataset_size=$2
run_id=$3
queries=$4

echo "Running TPCDS training"
echo "Following are the parameters:"
# echo "Data password: $data_password"
echo "Dataset size: $dataset_size"
echo "Run id: $run_id"
echo "Queries: $queries"

./prepare/prepare_tpcds.sh $data_password $dataset_size
python3 ./src/optimize_queries_training.py --size $dataset_size --id $run_id --dataset tpcds --queries $4
./teardown/teardown_tpcds.sh