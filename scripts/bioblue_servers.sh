#!/usr/bin/env bash
export MLFLOW_S3_ENDPOINT_URL=http://betelgeuse.elen.ucl.ac.be:9000
export MLFLOW_S3_IGNORE_TLS=true
export AK=vjoosdtb
export SK=qsdfghjk
export MINIO_ACCESS_KEY=$AK
export MINIO_SECRET_KEY=$SK
PYTHON_ENV=$(poetry env info --path)
source ${PYTHON_ENV}/bin/activate # /home/vjoosdeterbe/.cache/pypoetry/virtualenvs/bioblue-wqkgZuJr-py3.7/bin/activate

DIRECTORY=/data/workdir/vjoosdeterbe/bio-blueprints

tmux new-session -A -d -s monitor zsh
tmux setenv -t monitor MLFLOW_S3_ENDPOINT_URL $MLFLOW_S3_ENDPOINT_URL
tmux setenv -t monitor MLFLOW_S3_IGNORE_TLS $MLFLOW_S3_IGNORE_TLS
tmux setenv -t monitor AWS_ACCESS_KEY_ID $AK
tmux setenv -t monitor AWS_SECRET_ACCESS_KEY $SK
tmux setenv -t monitor MINIO_ACCESS_KEY $AK
tmux setenv -t monitor MINIO_SECRET_KEY $SK
tmux split-window -h -t monitor "mlflow server --backend-store-uri ${DIRECTORY}/mlruns --default-artifact-root s3://mlflow/ --host 0.0.0.0; sleep infinity"
echo "Launched mlflow server"
tmux split-window -h -t monitor:0 "${DIRECTORY}/minio server ${DIRECTORY}/artifacts; sleep infinity"
echo "Launched minio server"
tmux select-layout -t monitor even-horizontal
