wget https://github.com/sbt/sbt/releases/download/v1.9.9/sbt-1.9.9.tgz \
    && tar -xvf sbt-1.9.9.tgz \
    && rm sbt-1.9.9.tgz


cp -r ~/spark-tuning/docker/base_images/scripts ~/scripts

(cd ~/spark-listeners && exec ~/sbt/bin/sbt assembly)
(cd ~/tpcds-spark && exec ~/sbt/bin/sbt package)
(cd ~/tpch-spark && exec ~/sbt/bin/sbt package)