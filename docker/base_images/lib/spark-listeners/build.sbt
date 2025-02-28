name := "layer6-spark-listeners"
organization := "ai.layer6.spark.listeners"
version := "1.0"

scalaVersion := "2.12.18"

libraryDependencies += "ch.cern.sparkmeasure" %% "spark-measure" % "0.23"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.3" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.3" % "provided"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case _ => MergeStrategy.preferProject 
}