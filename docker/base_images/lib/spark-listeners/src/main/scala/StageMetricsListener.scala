package ai.layer6.spark.listeners

import org.apache.spark.scheduler.SparkListener
import ch.cern.sparkmeasure.StageInfoRecorderListener
import org.apache.spark.scheduler.SparkListenerApplicationEnd
import ch.cern.sparkmeasure.IOUtils
import scala.collection.mutable.ListBuffer
import ch.cern.sparkmeasure.StageVals
import org.apache.spark.SparkConf
import ch.cern.sparkmeasure.Utils
import scala.collection.mutable.{ListBuffer, LinkedHashMap}
import scala.math.{min, max}
import java.io.File
import java.io.FileWriter
import scala.collection.mutable.ArrayBuffer
import org.json4s._
import org.json4s.jackson.JsonMethods._
import scala.io.Source

case class StageMetricsOutput(applicationName: String, applicationId: String, stageMetricsData: ListBuffer[StageVals])

class StageMetricsListener(sc: SparkConf) extends StageInfoRecorderListener(false, Array.empty[String]) with ApplicationName {



    override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd) {
        val appName = AggregationUtils.applicationName(sc).getOrElse(applicationName)
        val appId = AggregationUtils.applicationId(sc).getOrElse(applicationId)

        // Write the raw metrics to a json file
        val output = StageMetricsOutput(appName, appId, stageMetricsData)
        val rawOutputFilename = s"${appName}_${appId}_stage_metrics.json"
        writeStageMetricsToJson(output, rawOutputFilename)


        // Write the metrics the aggregated metrics to a csv file
        val aggreagted_csv_filename = s"${appName}_${appId}_stage_metrics.csv"
        val aggregates_output = aggregateStageMetrics(stageMetricsData)
        writeStageMetricsToCSV(aggregates_output, aggreagted_csv_filename)
    }

    // Compute basic aggregation on the Stage metrics to save to a file
    def aggregateStageMetrics(stageMetricsData: ListBuffer[StageVals]) : LinkedHashMap[String, Long] = {

        val agg = Utils.zeroMetricsStage()
        var submissionTime = Long.MaxValue
        var completionTime = 0L

        for (metrics <- stageMetricsData){
            agg("numStages") += 1L
            agg("numTasks") += metrics.numTasks
            agg("stageDuration") += metrics.stageDuration
            agg("executorRunTime") += metrics.executorRunTime
            agg("executorCpuTime") += metrics.executorCpuTime
            agg("executorDeserializeTime") += metrics.executorDeserializeTime
            agg("executorDeserializeCpuTime") += metrics.executorDeserializeCpuTime
            agg("resultSerializationTime") += metrics.resultSerializationTime
            agg("jvmGCTime") += metrics.jvmGCTime
            agg("shuffleFetchWaitTime") += metrics.shuffleFetchWaitTime
            agg("shuffleWriteTime") += metrics.shuffleWriteTime
            agg("resultSize") = max(metrics.resultSize, agg("resultSize"))
            agg("diskBytesSpilled") += metrics.diskBytesSpilled
            agg("memoryBytesSpilled") += metrics.memoryBytesSpilled
            agg("peakExecutionMemory") += metrics.peakExecutionMemory
            agg("recordsRead") += metrics.recordsRead
            agg("bytesRead") += metrics.bytesRead
            agg("recordsWritten") += metrics.recordsWritten
            agg("bytesWritten") += metrics.bytesWritten
            agg("shuffleRecordsRead") += metrics.shuffleRecordsRead
            agg("shuffleTotalBlocksFetched") += metrics.shuffleTotalBlocksFetched
            agg("shuffleLocalBlocksFetched") += metrics.shuffleLocalBlocksFetched
            agg("shuffleRemoteBlocksFetched") += metrics.shuffleRemoteBlocksFetched
            agg("shuffleTotalBytesRead") += metrics.shuffleTotalBytesRead
            agg("shuffleLocalBytesRead") += metrics.shuffleLocalBytesRead
            agg("shuffleRemoteBytesRead") += metrics.shuffleRemoteBytesRead
            agg("shuffleRemoteBytesReadToDisk") += metrics.shuffleRemoteBytesReadToDisk
            agg("shuffleBytesWritten") += metrics.shuffleBytesWritten
            agg("shuffleRecordsWritten") += metrics.shuffleRecordsWritten
            submissionTime = min(metrics.submissionTime, submissionTime)
            completionTime = max(metrics.completionTime, completionTime)
        }
        agg("elapsedTime") = completionTime - submissionTime
        agg
    }
  
    // Writes the aggregated stage metrics to a CSV file. If the file already exists, the metrics are appended to the file.
    def writeStageMetricsToCSV(output: LinkedHashMap[String, Long], filename: String): Unit = {
        val file = new File(filename)

        if (file.exists() && !file.isDirectory()) {
            val fileWriter = new FileWriter(file, true)
            val rowData = output.values.toList
            val row = rowData.map(_.toString)
            fileWriter.append(row.mkString(","))
            fileWriter.append("\n")
            fileWriter.close()
        } else {
            val file = new File(filename)
            val fileWriter = new FileWriter(file, true)

            val headers = output.keys.toList
            val rows = ArrayBuffer(headers)

            val rowData = output.values.toList
            rows += rowData.map(_.toString)

            rows.foreach(row => {
                fileWriter.append(row.mkString(","))
                fileWriter.append("\n")
            })
            fileWriter.close()
        }
    }

    // Writes the stage metrics to a JSON file. If the file already exists, the metrics are appended to the file.
    def writeStageMetricsToJson(output: StageMetricsOutput, filename: String): Unit = {
        implicit val formats: DefaultFormats.type = DefaultFormats
        val file = new File(filename)
        if (file.exists() && !file.isDirectory()) {
            val existingJson = Source.fromFile(file).mkString
            val existingMetrics = parse(existingJson).extract[List[StageMetricsOutput]]
            val updatedMetrics = existingMetrics :+ output
            IOUtils.writeSerializedJSON(filename, updatedMetrics)
        } else {
            IOUtils.writeSerializedJSON(filename, List(output))
        }

    }

}


