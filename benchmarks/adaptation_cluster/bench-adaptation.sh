#!/bin/sh
set -e

cd $(dirname $0)
KUNGFU_ROOT=$(pwd)/../..

timeout=2m

cap=16
H=127.0.0.1:$cap

kungfu_run() {
    local init_np=$1
    shift
    ${KUNGFU_ROOT}/bin/kungfu-run \
        -H ${H} \
        -np $init_np \
        -timeout ${timeout} \
        -w \
        $@
}

export TF_CPP_MIN_LOG_LEVEL=0

# kungfu_run 2 python3 adaptive_trainer_tf2.py
# kungfu_run -np 2 python3 adaptive_trainer_tf2.py --schedule '1:16,1:1,1:16,1:1'
# kungfu_run 1 python3 adaptive_trainer_tf2.py --schedule '1:14,1:12,1:16,1:1'
kungfu_run 1 python3 adaptive_trainer_tf2.py --schedule '1:16,1:16,1:16,1:16,1:16,1:16,1:16,1:16,1:16,1:16,1:16,1:16'