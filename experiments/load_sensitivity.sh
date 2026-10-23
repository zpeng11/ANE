#!/bin/bash
# load_sensitivity.sh — measure ANE throughput with and without concurrent CPU load.
#
# Bounded smoke run: 3 x EVALS evals per phase, using the real NEON load generator
# (cpuhog) instead of `while :; do :; done`. Override EVALS/BLOCK/HOGS from the env.
#
# This harness measures throughput only. It does not establish a mechanism; any
# quantitative claim needs a controlled thermal/load setup.
set -o pipefail

cd "$(dirname "$0")" || exit 1

EVALS=${EVALS:-1200}
BLOCK=${BLOCK:-400}
HOGS=${HOGS:-8}
BUILD_DIR=${BUILD_DIR:-build}

case "$EVALS" in (*[!0-9]*|'') echo "EVALS must be a positive integer"; exit 1;; esac
case "$BLOCK" in (*[!0-9]*|'') echo "BLOCK must be a positive integer"; exit 1;; esac
case "$HOGS"  in (*[!0-9]*|'') echo "HOGS must be a positive integer"; exit 1;; esac
if [ "$EVALS" -le 0 ] || [ "$BLOCK" -le 0 ] || [ "$BLOCK" -gt "$EVALS" ]; then
    echo "need EVALS > 0 and 0 < BLOCK <= EVALS"
    exit 1
fi

for b in sustained cpuhog; do
    if [ ! -x "$BUILD_DIR/$b" ]; then
        echo "missing $BUILD_DIR/$b — build it first: make all"
        exit 1
    fi
done

pids=()
kill_hogs() {
    if [ ${#pids[@]} -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null
        wait "${pids[@]}" 2>/dev/null
        pids=()
    fi
}
trap kill_hogs EXIT INT TERM

run_sustained() {
    "$BUILD_DIR/sustained" 512 64 128 "$EVALS" "$BLOCK"
}

echo "########## A) baseline (no extra load) ##########"
run_sustained || { echo "A) sustained FAILED (exit $?)"; exit 1; }

echo
echo "########## B) WITH $HOGS CPU hogs ##########"
for i in $(seq 1 "$HOGS"); do
    "$BUILD_DIR/cpuhog" &
    pids+=($!)
done
sleep 2
run_sustained
rc=$?
kill_hogs
if [ "$rc" -ne 0 ]; then
    echo "B) sustained FAILED (exit $rc)"
    exit 1
fi

echo
echo "########## C) hogs killed, re-measure ##########"
sleep 3
run_sustained || { echo "C) sustained FAILED (exit $?)"; exit 1; }

echo "########## done ##########"
