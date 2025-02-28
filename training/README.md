# Training Data Collection

## TPC-H
Run from the `training` directory, not the subdirectories.

## Preparing Cluster
To prepare the cluster, run the `./prepare/prepare_tpch.sh` script with the required parameters

## Running Jobs and collecting data
TPC-H training data can be run by running the `tpch_training.py` script in `src`.

## Tearing down the cluster
To tear down the cluster, run `./teardown/teardown_tpch.sh`. Teardowns should be done after every application,
or after every single data size change.

## One-step command
To run all the optimizations for one dataset size do
`./run/run_tpch.sh <dataset_password> <dataset_size> <run_id>`

### Running in multi-machine mode
1. Create network `docker network create -d overlay cluster_net_swarm`
2. Add the following alias to your `.bashrc`: `alias swarmexec='function _swarmexec() { docker exec -it $(docker ps -q -f name="$1") "${@:2}"; }; _swarmexec'`