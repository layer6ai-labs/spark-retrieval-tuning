package ai.layer6.spark.listeners

import org.apache.spark.scheduler.SparkListener
import org.apache.spark.scheduler.SparkListenerApplicationStart
import org.apache.spark.scheduler.SparkListenerApplicationEnd
import org.apache.spark.SparkConf

object AggregationUtils {

    val APP_NAME_PROPERTY = "spark.listeners.layer6.appName"
    val APP_ID_PROPERTY = "spark.listeners.layer6.appId"
    val APP_START_TIME_PROPERTY = "spark.app.startTime"

    def applicationName(sparkConf: SparkConf): Option[String] = {sparkConf.getOption(APP_NAME_PROPERTY)}
    def applicationId(sparkConf: SparkConf): Option[String] = {sparkConf.getOption(APP_ID_PROPERTY)}
    def applicationStartTime(sparkConf: SparkConf): Option[String] = {sparkConf.getOption(APP_START_TIME_PROPERTY)}
}

trait ApplicationName extends SparkListener {
    var applicationName: String = _
    var applicationId: String = _

    override def onApplicationStart(appStart: SparkListenerApplicationStart) {
        applicationName = appStart.appName
        applicationId = appStart.appId.getOrElse("")
    }
}