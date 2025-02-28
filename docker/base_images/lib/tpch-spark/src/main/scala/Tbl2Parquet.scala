package main.scala
import org.apache.spark.sql.{DataFrame, SparkSession}
object Tbl2Parquet {
  def main(args: Array[String]): Unit = {

        // get paths from env variables else use default
        val cwd = System.getProperty("user.dir")
        val inputDataDir = sys.env.getOrElse("TPCH_INPUT_TBL_DIR", "file://" + cwd + "/dbgen")
        val outputDir = sys.env.getOrElse("TPCH_PARQUET_OUTPUT_DIR", inputDataDir + "/output")

        val spark = SparkSession
        .builder
        .appName("TPC-H v3.0.0 Spark Tbl2Parquet")
        .getOrCreate()
        val schemaProvider = new TpchSchemaProvider(spark, inputDataDir)
        import schemaProvider._
        customer.write.parquet(s"$outputDir/customer.parquet")
        lineitem.write.parquet(s"$outputDir/lineitem.parquet")
        nation.write.parquet(s"$outputDir/nation.parquet")
        region.write.parquet(s"$outputDir/region.parquet")
        order.write.parquet(s"$outputDir/order.parquet")
        part.write.parquet(s"$outputDir/part.parquet")
        partsupp.write.parquet(s"$outputDir/partsupp.parquet")
        supplier.write.parquet(s"$outputDir/supplier.parquet")
    }
}
