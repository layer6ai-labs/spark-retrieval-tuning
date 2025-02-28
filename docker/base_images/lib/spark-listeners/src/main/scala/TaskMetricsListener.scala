package ai.layer6.spark.listeners

import org.apache.spark.scheduler.SparkListener
import org.apache.spark.scheduler.SparkListenerApplicationEnd
import ch.cern.sparkmeasure.IOUtils
import scala.collection.mutable.ListBuffer
import ch.cern.sparkmeasure.TaskVals
import ch.cern.sparkmeasure.TaskInfoRecorderListener
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

case class TaskMetricsOutput(applicationName: String, applicationId: String, stageMetricsData: ListBuffer[TaskVals])

class TaskMetricsListener(sc: SparkConf) extends TaskInfoRecorderListener with ApplicationName{

    override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd) {
        val appName = AggregationUtils.applicationName(sc).getOrElse(applicationName)
        val appId = AggregationUtils.applicationId(sc).getOrElse(applicationId)

        // Write the raw metrics to a json file
        val output = TaskMetricsOutput(appName, appId, taskMetricsData)
        val rawOutputFilename = s"${appName}_${appId}_task_metrics.json"
        writeTaskMetricsToJson(output, rawOutputFilename)


        // Write the metrics the aggregated metrics to a csv file
        val aggreagted_csv_filename = s"${appName}_${appId}_task_metrics.csv"
        val aggregates_output = aggregateTaskMetrics(taskMetricsData)
        writeTaskMetricsToCSV(aggregates_output, aggreagted_csv_filename)
    }

    // Compute basic aggregation on the Task metrics for the metrics report
    // also filter on the time boundaries for the report
    def aggregateTaskMetrics(taskMetricsData: ListBuffer[TaskVals]) : LinkedHashMap[String, Long] = {

        val agg = Utils.zeroMetricsTask()

        for (metrics <- taskMetricsData){
            agg("numTasks") += 1L
            if (metrics.successful) {
                agg("successful tasks") += 1L
            }
            if (metrics.speculative) {
                agg("speculative tasks") += 1L
            }
            agg("taskDuration") += metrics.duration
            agg("schedulerDelayTime") += metrics.schedulerDelay
            agg("executorRunTime") += metrics.executorRunTime
            agg("executorCpuTime") += metrics.executorCpuTime
            agg("executorDeserializeTime") += metrics.executorDeserializeTime
            agg("executorDeserializeCpuTime") += metrics.executorDeserializeCpuTime
            agg("resultSerializationTime") += metrics.resultSerializationTime
            agg("jvmGCTime") += metrics.jvmGCTime
            agg("shuffleFetchWaitTime") += metrics.shuffleFetchWaitTime
            agg("shuffleWriteTime") += metrics.shuffleWriteTime
            agg("gettingResultTime") += metrics.gettingResultTime
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
        }
        agg
    }

    // Writes the task metrics to a CSV file. If the file already exists, the metrics are appended to the file.
    def writeTaskMetricsToCSV(output: LinkedHashMap[String, Long], filename: String): Unit = {
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

    def writeTaskMetricsToJson(output: TaskMetricsOutput, filename: String): Unit = {
        implicit val formats: DefaultFormats.type = DefaultFormats

        val file = new File(filename)
        if (file.exists() && !file.isDirectory()) {
            val existingJson = Source.fromFile(file).mkString
            val existingMetrics = parse(existingJson).extract[List[TaskMetricsOutput]]
            val updatedMetrics = existingMetrics :+ output
            IOUtils.writeSerializedJSON(filename, updatedMetrics)
        } else {
            IOUtils.writeSerializedJSON(filename, List(output))
        }

    }

    
}