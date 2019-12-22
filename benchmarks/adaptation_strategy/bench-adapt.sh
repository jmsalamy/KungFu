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

kungfu_run 16 python3 adaptive_strategy.py