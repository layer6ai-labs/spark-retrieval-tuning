REMOTE_HOSTNAME=layer6@192.168.4.16
REMOTE_MOUNT_SRC=/home/layer6
LOCAL_MOUNT_DST=/mnt/spark_tuning

if [ ! -d "$LOCAL_MOUNT_DST" ]; then
    mkdir "$LOCAL_MOUNT_DST"
fi
sshfs -o allow_other,default_permissions $REMOTE_HOSTNAME:$REMOTE_MOUNT_SRC $LOCAL_MOUNT_DST