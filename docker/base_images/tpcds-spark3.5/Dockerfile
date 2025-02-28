FROM spark-tuning-spark3.5:latest

COPY --from=lib /tpcds-spark /tpcds-spark

RUN cd /tpcds-spark && /sbt/bin/sbt package

COPY --from=scripts / /scripts