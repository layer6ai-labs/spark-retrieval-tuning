# Shell script for generating TPCDS data
set -exo pipefail

if [ -z "${SPARK_HOME}" ]; then
  echo "env SPARK_HOME not defined" 1>&2
  exit 1
fi

# Determine the current working directory
_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load common functions
. "${_DIR}/utils.sh"

# Resolve a jar location for the TPCDS data generator
find_resource() {
  local tpcds_spark_version=`grep "version" "${_DIR}/../build.sbt" | awk -F '\"' '{print $2}'`
  local scala_version=`grep "scalaVersion" "${_DIR}/../build.sbt" | awk -F '..\"' '{print $2}'`
  local jar_file="spark-tpc-ds-queries_${scala_version%?}-${tpcds_spark_version}.jar"
  local built_jar="$_DIR/../target/scala-${scala_version%?}/${jar_file}"
  echo $built_jar
  if [ -e "$built_jar" ]; then
    RESOURCE=$built_jar
  else
    RESOURCE="$_DIR/../assembly/${jar_file}"
    echo "${built_jar} not found, so use pre-compiled ${RESOURCE}" 1>&2
  fi
}

# If Spark not compiled, do it here
check_spark_compiled

# Do some preparations before launching spark-submit
parse_args_for_spark_submit "$@"
find_resource
find_output "$@"

# Complie tools in tpcds-kit
os_name="NOT_SUPPORT"
temp="$(uname -s)"
case $temp in
  "Linux")
    os_name="LINUX"
    echo $os_name
    ;;
  "Darwin")
    os_name="MACOS"
    echo $os_name
    ;;
  *)        ;;
esac
cd $_DIR/../tpcds-kit/tools
make OS=$os_name
cd $_DIR

echo "dsdgen generate data:"
$_DIR/../tpcds-kit/tools/dsdgen -DISTRIBUTIONS $_DIR/../tpcds-kit/tools/tpcds.idx -DIR $OUTPUT \
-SCALE $SCL


echo "Using \`spark-submit\` from path: $SPARK_HOME" 1>&2
exec "${SPARK_HOME}"/bin/spark-submit                           \
  --class cn.zyc.tpcdsspark.TpcdsDatagen3 \
  $(join_by " " ${SPARK_CONF[@]})                               \
  ${RESOURCE}                                                   \
  $(join_by " " ${ARGS[@]})

