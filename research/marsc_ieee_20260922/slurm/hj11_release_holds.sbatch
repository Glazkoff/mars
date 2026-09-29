#!/bin/bash
# H-J11 housekeeping: the gap-fill jobs were put ahead of the user's other pending jobs by holding those jobs
# (scontrol top is not permitted). This CPU-only job waits until the last H-J11 GPU job (passed as HJ11_LAST) has
# left the pending state, then releases every job listed in held_jobs.txt. Release is idempotent.
#SBATCH --job-name=mars-hj11-release
#SBATCH --partition=infer
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
L=/home/user/results/marsc_strengthen/hj11/held_jobs.txt
LAST="${HJ11_LAST:?set HJ11_LAST}"
# A failed squeue query must count as "still pending": an earlier version treated a transient error as "none
# pending" and released everything while gap-fill jobs were still queued.
still_pending () { local out; out=$(squeue -h -j "$LAST" -t PD -o %i 2>&1) || return 0; [ -n "$out" ]; }
while still_pending; do sleep 60; done
ids=$(sed "s/_.*//" "$L" | sort -u | tr "\n" ",")
scontrol release "${ids%,}"; echo "[hj11] released $(wc -l < "$L") held jobs at $(date -u)"
