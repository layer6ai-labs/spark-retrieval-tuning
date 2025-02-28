FROM spark-tuning-base:latest

# install spark
RUN wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz \
    && tar -xvf spark-3.5.1-bin-hadoop3.tgz \
    && mv spark-3.5.1-bin-hadoop3 /opt/spark \
    && rm spark-3.5.1-bin-hadoop3.tgz

ENV SPARK_HOME="/opt/spark"

COPY --from=lib /spark-listeners lib/spark-listeners/

RUN cd lib/spark-listeners && /sbt/bin/sbt assembly

ENV SPARK_LISTENERS_JAR="/lib/spark-listeners/target/scala-2.12/layer6-spark-listeners-assembly-1.0.jar"
ENV SPARK_EXTRA_LISTENERS="ai.layer6.spark.listeners.TaskMetricsListener,ai.layer6.spark.listeners.StageMetricsListener"
ENV SPARK_QUERY_LISTENERS="ai.layer6.spark.listeners.QueryPlanListener"
ENV SPARK_WORKER_OPTS="$SPARK_WORKER_OPTS -Dspark.worker.cleanup.enabled=true -Dspark.worker.cleanup.appDataTtl=1800"