FROM spark-tuning-spark3.5:latest

ENV SPARK_MASTER="spark://spark-master:7077"

# download tpc-h
COPY --from=lib /tpch-spark /tpch-spark

# build tpc-h queries
WORKDIR /tpch-spark
RUN /sbt/bin/sbt package

COPY --from=scripts / /scripts

WORKDIR /