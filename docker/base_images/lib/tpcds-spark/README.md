# tpcds-spark
TPC-DS queries in spark SQL

This project gains inspirations from [ssavvides/tpch-spark](https://github.com/ssavvides/tpch-spark) and the implementation is based on [maropu/spark-tpcds-datagen](https://github.com/maropu/spark-tpcds-datagen).



## Usage

### Setup
- Setup [Spark](https://spark.apache.org/) 2.4.0
- Setup SPARK_HOME:`export SPARK_HOME=<path_to_spark>`
- DownLoad and package tpcds-spark
	```bash
	git clone https://github.com/SimonZYC/tpcds-spark.git

	cd /<path_to>/tpcds-spark/

	sbt package
	```

- In `$SPARK_HOME/conf/spark-defaults.conf` ( If you do not have it, `cp spark-defaults.conf.template spark-defaults.conf `), add this:

  ```
  spark.sql.crossJoin.enabled		true
  ```

  

### Data generation

```bash
# We must execute programs in ubin directory
cd /<path_to>/tpcds-spark/ubin
./dsdgen.sh --output-location ../data --scale-factor 5

# For more options:
./dsdgen.sh --help
Usage: spark-submit --class <this class> --conf key=value <spark tpcds datagen jar> [Options]
Options:
  --output-location [STR]                Path to an output location
  --scale-factor [NUM]                   Scale factor (default: 1)
  --format [STR]                         Output format (default: parquet)
  --overwrite                            Whether it overwrites existing data (default: false)
  --partition-tables                     Whether it partitions output data (default: false)
  --use-double-for-decimal               Whether it prefers double types (default: false)
  --cluster-by-partition-columns         Whether it cluster output data by partition columns (default: false)
  --filter-out-null-partition-values     Whether it filters out NULL partitions (default: false)
  --table-filter [STR]                   Queries to filter, e.g., catalog_sales,store_sales
  --num-partitions [NUM]                 # of partitions (default: 100)
```

There are 3 Tpcds data generation classes( objects ) and 3 scripts( `dsdgen.sh, dsdgen2.sh, dsdgen3.sh `).
- `dsdgen.sh` (`TPCDSDatagen.scala`) are from [maropu/spark-tpcds-datagen](https://github.com/maropu/spark-tpcds-datagen) and they suppory all the options above. The drawback is that you need huge memory when the scale is high, because it loads 1 whole table into memory each time.
- `dsdgen2.sh` (`TpcdsDatagen2.scala, TpcdsSchemaProvider.scala`) are based on [ssavvides/tpch-spark](https://github.com/ssavvides/tpch-spark). The queries on data they generate may not produce correct answer.
- `dsdgen3.sh` (`TpcdsDatagen3.scala`) are combinations of the above 2. However, they only support `--output-location` and `--scale-factor` now.

`dsdgen3.sh` is most recommended.

```bash
cd /<path_to>/tpcds-spark/ubin
./dsdgen.sh --output-location ../data --scale-factor 5
```



### Run All TPC-DS Queries

```bash
# We must execute programs in ubin directory
cd /<path_to>/tpcds-spark/ubin

./run-tpcs.sh --data-location [TPC-DS test data]

# We could add Spark arguments:
./run-tpcs.sh --data-location [TPC-DS test data] --master local[*]
```

### Run Specific TPC-DS Queries

```bash
# We must execute programs in ubin directory
cd /<path_to>/tpcds-spark/ubin

./run-tpcds.sh --data-location [TPC-DS test data] --query-filter "q2,q10"
```

Notice: There are queries which have the same names in V1.4 and V2.7. You could take a look at [reources](src/main/resources).

If you want to execute queries which appear in both V1.4 and V2.7, you could use "q11-27" to show it is "q11" in V2.7, or "q11-14" meaning "q11" in V1.4

```
./run-tpcds.sh --data-location [TPC-DS test data] --query-filter "q11-27"
```



## Reference

[ssavvides/tpch-spark](https://github.com/ssavvides/tpch-spark)

[maropu/spark-tpcds-datagen](https://github.com/maropu/spark-tpcds-datagen)

[tpcds-kit](<https://github.com/databricks/tpcds-kit>)

[tpcds](http://www.tpc.org/tpcds/)