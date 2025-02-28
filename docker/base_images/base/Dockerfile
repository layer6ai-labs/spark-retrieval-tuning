FROM ubuntu:latest


USER root

RUN apt-get update -y \
    && apt-get install wget -y \
    && apt-get install ssh -y \
    && apt-get install python3 -y \
    && apt-get install vim -y \
    && apt-get install git -y \
    && apt-get install make -y \
    && apt-get install bison -y \
    && apt-get install byacc -y \
    && apt-get install flex -y \
    && apt-get install sshpass -y 

# Install GCC 9
RUN apt-get update && \
    apt-get install -y gcc-9

# Create a symbolic link to gcc-9 as gcc
RUN ln -s /usr/bin/gcc-9 /usr/local/bin/gcc

# Set gcc as the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100

# install jdk 8
RUN apt-get install openjdk-8-jdk -y 

ENV JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-amd64"

# install maven
RUN wget https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz \
    && tar -xvf apache-maven-3.9.6-bin.tar.gz \
    && mv apache-maven-3.9.6 /opt/maven \
    && rm apache-maven-3.9.6-bin.tar.gz

ENV PATH="${PATH}:/opt/maven/bin"

# install scala
RUN wget https://downloads.lightbend.com/scala/2.12.18/scala-2.12.18.deb \
    && dpkg -i scala-2.12.18.deb \
    && apt-get install scala \
    && rm scala-2.12.18.deb

# install hadoop
RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.2.4/hadoop-3.2.4.tar.gz \
    && tar -xvf hadoop-3.2.4.tar.gz \ 
    && mv hadoop-3.2.4 /opt/hadoop \
    && rm hadoop-3.2.4.tar.gz

RUN echo "export JAVA_HOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64'" >> "/opt/hadoop/etc/hadoop/hadoop-env.sh"
COPY core-site.xml /opt/hadoop/etc/hadoop/
COPY hdfs-site.xml /opt/hadoop/etc/hadoop/

ENV PATH="${PATH}:/opt/hadoop/bin"
ENV HADOOP_HOME="/opt/hadoop"

ENV HDFS_NAMENODE_USER=root
ENV HDFS_DATANODE_USER=root
ENV HDFS_SECONDARYNAMENODE_USER=root
ENV YARN_RESOURCEMANAGER_USER=root
ENV YARN_NODEMANAGER_USER=root

# setup ssh so hadoop can start a single node server
RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa \
    && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys \
    && chmod 0600 ~/.ssh/authorized_keys

# download sbt
RUN wget https://github.com/sbt/sbt/releases/download/v1.9.9/sbt-1.9.9.tgz \
    && tar -xvf sbt-1.9.9.tgz \
    && rm sbt-1.9.9.tgz