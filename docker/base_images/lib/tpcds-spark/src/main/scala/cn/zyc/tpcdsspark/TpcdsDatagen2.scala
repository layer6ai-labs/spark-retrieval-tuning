package cn.zyc.tpcdsspark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import java.io.File

object TpcdsDatagen2 {
  def main(args: Array[String]): Unit = {
    val datagenArgs = new TPCDSDatagenArguments(args)
    val spark = SparkSession.builder.appName("TPC-DS Data Generations").getOrCreate()

    val sc = spark.sparkContext

    val schemaProvider = new TpcdsSchemaProvider(sc, datagenArgs.outputLocation)
    import schemaProvider._
    for(i <- 0 until table_list.size){
      schemaProvider.table_list(i).write.format("parquet").save(s"${datagenArgs.outputLocation}/${schemaProvider.table_names(i)}")
      val path: File = new File(s"${datagenArgs.outputLocation}/${schemaProvider.table_names(i)}.dat")
      if (path.isFile()) {
        path.delete()
      }
    }
//
    //    tpcdsTables.genData(
//      datagenArgs.outputLocation,
//      datagenArgs.format,
//      datagenArgs.overwrite,
//      datagenArgs.partitionTables,
//      datagenArgs.useDoubleForDecimal,
//      datagenArgs.clusterByPartitionColumns,
//      datagenArgs.filterOutNullPartitionValues,
//      datagenArgs.tableFilter,
//      datagenArgs.numPartitions.toInt)
  }
}
