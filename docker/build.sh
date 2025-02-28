#!/usr/bin/env bash
BUILD_ROOT=$(dirname "$0")
BUILD_CONTEXT="$BUILD_ROOT/base_images"
LIB_CONTEXT="$BUILD_CONTEXT/lib"
SCRIPT_CONTEXT="$BUILD_CONTEXT/scripts"

function build {
    docker build -t spark-tuning-base:latest ${BUILD_CONTEXT}/base

    case $1 in

        base)
            ;;

        spark35)
            docker buildx build --build-context lib=$LIB_CONTEXT -t spark-tuning-spark3.5:latest ${BUILD_CONTEXT}/spark3.5
            ;;

        *)
            docker buildx build --build-context scripts=$SCRIPT_CONTEXT --build-context lib=$LIB_CONTEXT -t spark-tuning-spark3.5:latest ${BUILD_CONTEXT}/spark3.5

            docker buildx build --build-context scripts=$SCRIPT_CONTEXT --build-context lib=$LIB_CONTEXT -t spark-tuning-spark3.5-tpcds:latest ${BUILD_CONTEXT}/tpcds-spark3.5

            docker buildx build --build-context scripts=$SCRIPT_CONTEXT --build-context lib=$LIB_CONTEXT -t spark-tuning-spark3.5-tpch:latest ${BUILD_CONTEXT}/tpch-spark3.5

            ;;
    esac
}

build $1