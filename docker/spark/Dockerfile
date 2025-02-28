FROM spark:3.5.0-scala2.12-java17-ubuntu

ARG MAVEN_DOWNLOAD_LINK=https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz

USER root

RUN mkdir /usr/share/maven \
    && curl -fsSL -o /tmp/apache-maven.tar.gz ${MAVEN_DOWNLOAD_LINK} \
    && tar -xzf /tmp/apache-maven.tar.gz -C /usr/share/maven \
    && ln -s /usr/share/maven/apache-maven-3.9.6/bin/mvn /usr/bin/mvn

RUN mkdir hibench_repo 

RUN set -ex; \
    apt-get update; \
    apt-get install -y python3 python3-pip; \
    rm -rf /var/lib/apt/lists/*

USER root