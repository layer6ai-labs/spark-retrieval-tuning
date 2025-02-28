package ai.layer6.spark.listeners

import org.apache.spark.sql.util.QueryExecutionListener
import org.apache.spark.sql.execution.QueryExecution
import ch.cern.sparkmeasure.IOUtils

import java.io.File
import java.io.FileWriter
import scala.collection.mutable.ArrayBuffer
import org.json4s._
import org.json4s.jackson.JsonMethods._
import scala.io.Source

case class QueryPlanOutput(
  applicationName: String,
  applicationStartTime: String, 
  logicalPlan: String, 
  optimizedPlan: String, 
  sparkPlan: String,
  physicalPlan: String, 
  durationMs: Long,
  status: String,
  errorMessage: String = "")


// The AppPlanOutput is a list of query plans for a given application run
case class AppPlanOutput(
  applicationName: String,
  applicationStartTime: String, 
  queryPlans: List[QueryPlanOutput])


class QueryPlanListener extends QueryExecutionListener {

  override def onFailure(funcName: String, qe: QueryExecution, exception: Exception) {
    saveMetrics(funcName, qe, 0, "FAILURE", exception.getMessage)
  }

  override def onSuccess(funcName: String, qe: QueryExecution, durationNs: Long) {
    saveMetrics(funcName, qe, durationNs, "SUCCESS")
  }

  def saveMetrics(funcName: String, qe: QueryExecution, durationNs: Long, status: String, errorMessage: String = ""){
    val appName = AggregationUtils.applicationName(qe.sparkSession.sparkContext.getConf).getOrElse("applicationName")
    val appId = AggregationUtils.applicationId(qe.sparkSession.sparkContext.getConf).getOrElse("")
    val appStartTime = AggregationUtils.applicationStartTime(qe.sparkSession.sparkContext.getConf).getOrElse("")
    

    val filename = s"${appName}_${appId}_query_plans.json"
    
    val queryOut = QueryPlanOutput(
      applicationName = funcName,
      applicationStartTime = appStartTime,
      logicalPlan = qe.logical.toString,
      optimizedPlan = qe.optimizedPlan.toString,
      sparkPlan = qe.sparkPlan.toString,
      physicalPlan = qe.executedPlan.toString,
      durationMs = durationNs / 1000000,
      status = status,
      errorMessage = errorMessage
    )

    // Want to save the query plan to a json file
    writePlanToJson(queryOut, filename)
  }

  // Write the query plan to a json file. If the file already exists, append the query plan to the existing file
  def writePlanToJson(queryPlan: QueryPlanOutput, filename: String): Unit = {
    implicit val formats: DefaultFormats.type = DefaultFormats

    // There are three cases to consider:
    // 1. The file does not exist. So we create a new file and write the application plan with the new query plan to it.
    // 2. The file exists and the application name but this is a new run of the application. 
    //    So we append the query plan to the existing file as a new application plan
    // 3. The file exists and the application name and the application has already been run, so 
    //   we append the query plan to the existing application plan

    val file = new File(filename)
    if (file.exists() && !file.isDirectory()) {
        val existingJson = Source.fromFile(file).mkString
        val existingPlans = parse(existingJson).extract[List[AppPlanOutput]]
        var planExists = false

        // Loop through existing plans and append queryPlan if StartTime matches
        val queryPlans:List[AppPlanOutput] = existingPlans.map { appPlan =>
          if (appPlan.applicationStartTime == queryPlan.applicationStartTime) {
            planExists = true
            appPlan.copy(queryPlans = appPlan.queryPlans :+ queryPlan)
          } else {
            appPlan
          }
        }

        
        var updatedPlans = List[AppPlanOutput]()
        if (!planExists){
          updatedPlans = existingPlans :+ AppPlanOutput(
            applicationName = queryPlan.applicationName,
            applicationStartTime = queryPlan.applicationStartTime,
            queryPlans = List(queryPlan))
        }else{
           updatedPlans = queryPlans
        } 

        IOUtils.writeSerializedJSON(filename, updatedPlans)

    } else {
      val plan = AppPlanOutput(
        applicationName = queryPlan.applicationName,
        applicationStartTime = queryPlan.applicationStartTime,
        queryPlans = List(queryPlan))
      IOUtils.writeSerializedJSON(filename, List(plan))
    }
  }

}