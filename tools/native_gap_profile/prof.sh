#!/usr/bin/env bash
# prof.sh TAG LAUNCH_LIMIT KERNEL_FILTER -- command...
# Profiles one compression call with Nsight Compute (caches flushed per kernel).
set -u
TAG=$1; LIM=$2; KF=$3; shift 4
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
cd "$(dirname "$0")"
KARG=(); [ "$KF" != "-" ] && KARG=(-k "regex:$KF")
sudo env FZ_SPECIALIZE=${FZ_SPECIALIZE:-auto} /usr/local/cuda-12.9/bin/ncu --metrics $M "${KARG[@]}" -c $LIM --csv --page raw --log-file ncu_$TAG.csv "$@" > run_$TAG.log 2>&1
echo "$TAG rc=$? kernels=$(($(grep -c '^"' ncu_$TAG.csv 2>/dev/null)-2))"
