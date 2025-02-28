package cn.zyc.tpcdsspark

import java.io.{InputStream, BufferedWriter, File, FileWriter, PrintWriter}
import org.apache.spark.sql._

import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.SparkSession

/**
 * Parent class for TPC-DS queries.
 *
 * 
 *
 * Yunchuan Zheng <yczheng1997@gmail.com>
 *
 */

object TpcdsQuery {

  val tables = Seq("catalog_page", "catalog_returns", "customer", "customer_address",
    "customer_demographics", "date_dim", "household_demographics", "inventory", "item",
    "promotion", "store", "store_returns", "catalog_sales", "web_sales", "store_sales",
    "web_returns", "web_site", "reason", "call_center", "warehouse", "ship_mode", "income_band",
    "time_dim", "web_page")

  
  val spark = SparkSession
          .builder()
          .appName("TPC-DS Query")
          .getOrCreate()

  def dropDuplicate(df: DataFrame): DataFrame = {
    val oriCol = df.columns
    val dis = oriCol.distinct
    val dup = oriCol.diff(dis)
    val recordMap = scala.collection.mutable.Map.empty[String, Int]
    for(ele <- dup){
      recordMap += (ele -> 0)
    }
    for(i <- oriCol.indices){
      if(recordMap.contains(oriCol(i)) && recordMap(oriCol(i)) == 0){
        recordMap(oriCol(i)) = 1
        oriCol(i) = oriCol(i) + "--suffix--"
      }
    }
    val dfNew = df.toDF(oriCol:_*)
    dfNew.drop(dup:_*).toDF(dis:_*)
  }

  def outputDF(df: DataFrame, outputDir: String, className: String): Unit = {

    if (outputDir == null || outputDir == "")
      df.collect().foreach(println)
    else
      //df.write.mode("overwrite").json(outputDir + "/" + className + ".out") // json to avoid alias

      df.write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").save(outputDir + "/" + className)
  }


  def runTpcdsQueries(
      queryLocation: String,
      queries: Seq[String],
      nameSuffix: String = ""
      ): ListBuffer[(String, Float)] ={
    
    val OUTPUT_DIR: String = "file://" + new File(".").getAbsolutePath() + "/../output_result/output"

    val results = new ListBuffer[(String, Float)]
    if(queries == null || queries.length == 0)
      return results

    for(i <- 0 until queries.length){
      val t0 = System.nanoTime()
      val name = queries(i)
      
      // val ou = new PrintWriter("/Users/zyc/Documents/code/tpcds-spark/debug.out")
      // ou.println(s"$queryLocation/$name.sql")
      // ou.println(getClass)
      // ou.println(getClass.getResource(s"/"))
      // ou.println(getClass.getClassLoader.getResource(""))
      // ou.println(new File(".").getAbsolutePath()+"/../src/main/resources")
      // ou.close()
      
      import scala.io.Source
      val sqlFile = Source.fromFile(new File(".").getAbsolutePath()+"/../src/main/resources"
      + s"/$queryLocation/$name.sql")
      val queryString = sqlFile.mkString

//      val q = spark.sql(queryString)
//      val q1 = dropDuplicate(q)
//      val ou = new PrintWriter("/Users/zyc/Documents/code/tpcds-spark/debug.out")
//      ou.println(q.columns.toList)
//      ou.println(q1.columns.toList)
//      ou.close()
      outputDF(dropDuplicate(spark.sql(queryString)), OUTPUT_DIR, s"$name$nameSuffix")

      val t1 = System.nanoTime()

      val elapsed = (t1 - t0) / 1000000000.0f // second
      results += new Tuple2(name, elapsed)
    }
     return results
  }

  def filterQueries(
      origQueries: Seq[String],
      queryFilter: Set[String], isFilter:Boolean): Seq[String] = {
    if (isFilter) {
      origQueries.filter(queryFilter.contains)
    } else {
      origQueries
    }
  }

  def setupTables(dataLocation: String): Unit = {
    tables.foreach { tableName =>
      spark.read.parquet(s"$dataLocation/$tableName").createOrReplaceTempView(tableName)
      tableName -> spark.table(tableName).count()
    }
  }

  def main(args: Array[String]): Unit = {

    val tpcdsArgs = new TPCDSQueryArguments(args)

    // List of all TPC-DS v1.4 queries
    val tpcdsQueries = Seq(
      "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
      "q12", "q13", "q14a", "q14b", "q15", "q16", "q17", "q18", "q19", "q20",
      "q21", "q22", "q23a", "q23b", "q24a", "q24b", "q25", "q26", "q27", "q28", "q29", "q30",
      "q31", "q32", "q33", "q34", "q35", "q36", "q37", "q38", "q39a", "q39b", "q40",
      "q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50",
      "q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58", "q59", "q60",
      "q61", "q62", "q63", "q64", "q65", "q66", "q67", "q68", "q69", "q70",
      "q71", "q72", "q73", "q74", "q75", "q76", "q77", "q78", "q79", "q80",
      "q81", "q82", "q83", "q84", "q85", "q86", "q87", "q88", "q89", "q90",
      "q91", "q92", "q93", "q94", "q95", "q96", "q97", "q98", "q99")

    // This list only includes TPC-DS v2.7 queries that are different from v1.4 ones
    val tpcdsQueriesV2_7 = Seq(
      "q5a", "q6", "q10a", "q11", "q12", "q14", "q14a", "q18a",
      "q20", "q22", "q22a", "q24", "q27a", "q34", "q35", "q35a", "q36a", "q47", "q49",
      "q51a", "q57", "q64", "q67a", "q70a", "q72", "q74", "q75", "q77a", "q78",
      "q80a", "q86a", "q98")

    // If `--query-filter` defined, filters the queries that this option selects
    val queriesV1_4ToRun = filterQueries(tpcdsQueries, tpcdsArgs.query14Filter, tpcdsArgs.isFilter)
    val queriesV2_7ToRun = filterQueries(tpcdsQueriesV2_7, tpcdsArgs.query27Filter, tpcdsArgs.isFilter)

    if ((queriesV1_4ToRun ++ queriesV2_7ToRun).isEmpty) {
      throw new RuntimeException(
        s"Empty queries to run. Bad query name filter: v1.4: ${tpcdsArgs.query14Filter}\n" +
          s"v2.7: ${tpcdsArgs.query27Filter}")
    }
    
    val tableSizes = setupTables(tpcdsArgs.dataLocation)

    val output = new ListBuffer[(String, Float)]

    output ++= runTpcdsQueries(queryLocation = "tpcds", queries = queriesV1_4ToRun, "")
    output ++= runTpcdsQueries(queryLocation = "tpcds-v2.7.0", queries = queriesV2_7ToRun, nameSuffix = "-v2.7")

    val outFile = new File("../output_result/TIMES.txt")
    val bw = new BufferedWriter(new FileWriter(outFile, true))

    output.foreach {
      case (key, value) => bw.write(f"${key}%s\t${value}%1.8f\n")
    }

    bw.close()
  }
}
