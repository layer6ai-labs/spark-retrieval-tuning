package main.scala

import org.apache.spark.sql.{DataFrame, SparkSession}

class TpchParquetSchemaProvider(spark: SparkSession, inputDir: String) {
    import spark.implicits._
   
    val dfMap = Map(
        "customer" -> spark.read.parquet(inputDir + "/customer.parquet"),
        "lineitem" -> spark.read.parquet(inputDir + "/lineitem.parquet"),
        "nation" -> spark.read.parquet(inputDir + "/nation.parquet"),
        "region" -> spark.read.parquet(inputDir + "/region.parquet"),
        "order" -> spark.read.parquet(inputDir + "/order.parquet"),
        "part" -> spark.read.parquet(inputDir + "/part.parquet"),
        "partsupp" -> spark.read.parquet(inputDir + "/partsupp.parquet"),
        "supplier" -> spark.read.parquet(inputDir + "/supplier.parquet")
    )

    dfMap.foreach {
        case (key, value) => value.createOrReplaceTempView(key)
    }

    val customer = dfMap.get("customer").get
    val lineitem = dfMap.get("lineitem").get
    val nation = dfMap.get("nation").get
    val region = dfMap.get("region").get
    val order = dfMap.get("order").get
    val part = dfMap.get("part").get
    val partsupp = dfMap.get("partsupp").get
    val supplier = dfMap.get("supplier").get
}
