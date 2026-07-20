#!/usr/bin/env bash
# Launches the SMRD/DBPD mechanism-decomposition experiment matrix
# (claude_code_experiment_spec.md Sec 3) via main.py, in the required
# priority order:
#   1. baseline, dbpd, smrd   (all seeds - replication gate)
#   2. delayed_dbpd
#   3. soft_dbpd
#   4. combined
#
# Usage:
#   ./launch_all.sh                     # run the full matrix
#   ./launch_all.sh --dry-run           # print every resolved command, run nothing
#   ./launch_all.sh --force             # re-run conditions whose output already exists
#
# delayed_dbpd and combined need an onset (t0). The spec gives this NO
# default (Sec 9.2: "Morgan will supply value(s)") - you must set one:
#   ONSET=5 ./launch_all.sh             # single onset
#   ONSETS="3 5 10" ./launch_all.sh     # small grid - each value becomes its
#                                        # own condition dir, e.g. delayed_dbpd_onset5
# Without either set, priorities 2 and 4 are skipped with a warning; 1 and 3
# still run.
#
# Escalating to 5 seeds (spec Sec 3): SEEDS="0 1 2 3 4" ./launch_all.sh
# Running a subset of conditions: CONDITIONS="baseline dbpd cp_dbpd" ./launch_all.sh
#   (space-separated condition dir names; unset/empty = run everything below)
# Single fixed seed for a quick pass: SEEDS=42 ./launch_all.sh
#
# cp_dbpd (critical_periods_dbpd) is included here too even though it isn't
# one of the spec's C1-C6 - it's this codebase's other DBPD-adjacent method
# (gradient-confusion-detected phase switch instead of a fixed onset).
#
# Adding a new condition: add another run_condition block following the
# pattern below - one line per condition, in priority order.
#
# A single condition crashing (segfault, OOM, driver hiccup) does NOT abort
# the rest of the matrix: each run is retried up to MAX_RETRIES times
# (default 2) and, if still failing, logged to a final failure summary while
# the script moves on to the next condition. Override with e.g.
# MAX_RETRIES=1 ./launch_all.sh to disable retries.
#
# NOT implemented here (see README/conversation for the full gap list):
# mid-run checkpoint/resume (a failed run restarts from epoch 0, not where it
# crashed), NaN/divergence abort, disk-space guard, smoke-test mode,
# git-dirty-tree guard. The only "already done" safety net is the
# completed-run skip below, which is a best-effort proxy (checks for the
# dynamics plot PNG, the last file MetricsRecorder writes on a successful
# run) - not a true final_summary.json "status": "completed" check, since
# main.py doesn't
# write one yet.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Deliberately NOT `set -e`: one condition crashing (segfault, OOM, driver
# hiccup) must not take the rest of the matrix down with it. Each run's exit
# code is checked explicitly in run_condition below and failures are
# collected into FAILED_CONDITIONS for a summary at the end, instead of
# aborting the whole script.

# ---------------------------------------------------------------------------
# Fixed training setup (spec Sec 1) - identical across every condition.
# ---------------------------------------------------------------------------
DATASET="cifar10"
EPOCHS=30
BATCH_SIZE=32
MODEL="${MODEL:-mobilenet_v2}"   # spec Sec 9.1: "confirm choice with Morgan" - defaulted
                                  # to what's been used throughout this project so far;
                                  # override with MODEL=... if that's not the final choice.
TASK="classification"
TAU="${TAU:-0.3}"                # spec Sec 9.4: primary threshold. 0.7 is a second-wave option.
KEEP_RATE="${KEEP_RATE:-0.1}"    # spec Sec 9.3 default for soft_dbpd/combined.
FINAL_REVISION_EPOCHS="${FINAL_REVISION_EPOCHS:-1}"

# Seeds (spec Sec 3): 3 minimum. Bump to "0 1 2 3 4" for the 5-seed escalation,
# or override entirely, e.g. SEEDS=42 for a single quick pass.
SEEDS="${SEEDS:-0 1 2}"

# Optional condition filter - space-separated condition dir names. Empty/unset
# runs every condition below (the default, full-matrix behaviour).
CONDITIONS="${CONDITIONS:-}"

ONSETS="${ONSETS:-${ONSET:-}}"
read -ra ONSET_ARR <<< "$ONSETS"
ONSET_COUNT=${#ONSET_ARR[@]}

PYTHON="${PYTHON:-python}"
RESULTS_ROOT="results"
MAX_RETRIES="${MAX_RETRIES:-2}"   # per-condition retry count on a nonzero exit (e.g. a transient segfault)

condition_enabled() {
    [[ -z "$CONDITIONS" ]] && return 0
    [[ " $CONDITIONS " == *" $1 "* ]]
}

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

run_count=0
skip_count=0
FAILED_CONDITIONS=()

# run_condition <condition_dir_name> <seed> <extra main.py args...>
run_condition() {
    local condition="$1" seed="$2"
    shift 2

    if ! condition_enabled "$condition"; then
        return
    fi

    local save_dir="${RESULTS_ROOT}/${DATASET}/${condition}/seed${seed}"
    local save_path="${save_dir}/${MODEL}"
    # "${save_path}_test_epochs_*.png" (glob) is the completion marker: it's
    # the last file written by every mode, MetricsRecorder-based (baseline,
    # dbpd, cp_dbpd, soft_dbpd, delayed_dbpd, combined) or not
    # (train_with_random/smrd, which never writes a MetricsRecorder
    # "_dynamics.png" at all - that file alone is NOT a reliable marker,
    # it'll silently never detect smrd as complete).
    local done_markers=("${save_path}"_test_epochs_*.png)

    if [[ -e "${done_markers[0]}" && "$FORCE" -eq 0 ]]; then
        echo "[skip] ${condition} seed=${seed} - already completed (${done_markers[0]} exists; use --force to re-run)"
        skip_count=$((skip_count + 1))
        return
    fi

    mkdir -p "$save_dir"
    local cmd=("$PYTHON" main.py --model "$MODEL" --task "$TASK" --dataset "$DATASET" \
        --epoch "$EPOCHS" --batch_size "$BATCH_SIZE" --seed "$seed" \
        --save_path "$save_path" "$@")

    run_count=$((run_count + 1))
    printf '[%02d] %s\n' "$run_count" "${cmd[*]}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi

    local attempt=1
    while true; do
        if "${cmd[@]}"; then
            return
        fi
        local exit_code   # split from assignment - `local x=$?` masks the real exit code (shellcheck SC2155)
        exit_code=$?
        if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
            echo "[FAILED] ${condition} seed=${seed} - exit code ${exit_code} after ${attempt} attempt(s), giving up on this condition and continuing with the rest" >&2
            FAILED_CONDITIONS+=("${condition} seed=${seed} (exit ${exit_code})")
            return
        fi
        attempt=$((attempt + 1))
        echo "[retry] ${condition} seed=${seed} - exit code ${exit_code}, attempt ${attempt}/${MAX_RETRIES}" >&2
    done
}

echo "== Priority 1: baseline, dbpd, smrd (replication gate) =="
for seed in $SEEDS; do
    run_condition "baseline" "$seed" --mode baseline
done
for seed in $SEEDS; do
    run_condition "dbpd" "$seed" --mode train_with_revision --threshold "$TAU" --start_revision $((EPOCHS - 1))
done
for seed in $SEEDS; do
    run_condition "smrd" "$seed" --mode train_with_random --threshold "$TAU" --start_revision $((EPOCHS - 1))
done

echo "== cp_dbpd (not part of the spec's C1-C6, run alongside the matrix) =="
for seed in $SEEDS; do
    run_condition "cp_dbpd" "$seed" --mode critical_periods_dbpd --threshold "$TAU" \
        --final_revision_epochs "$FINAL_REVISION_EPOCHS"
done

echo "== Priority 2: delayed_dbpd =="
if [[ "$ONSET_COUNT" -eq 0 ]]; then
    echo "ONSET/ONSETS not set - skipping delayed_dbpd (spec Sec 9.2: no default, must be supplied)." >&2
    echo "  e.g.: ONSET=5 ./launch_all.sh          (single value)" >&2
    echo "        ONSETS=\"3 5 10\" ./launch_all.sh  (grid)" >&2
else
    for onset in "${ONSET_ARR[@]}"; do
        cond="delayed_dbpd"
        [[ "$ONSET_COUNT" -gt 1 ]] && cond="delayed_dbpd_onset${onset}"
        for seed in $SEEDS; do
            run_condition "$cond" "$seed" --mode delayed_dbpd --threshold "$TAU" \
                --onset "$onset" --final_revision_epochs "$FINAL_REVISION_EPOCHS"
        done
    done
fi

echo "== Priority 3: soft_dbpd =="
for seed in $SEEDS; do
    run_condition "soft_dbpd" "$seed" --mode soft_dbpd --threshold "$TAU" \
        --keep_rate "$KEEP_RATE" --final_revision_epochs "$FINAL_REVISION_EPOCHS"
done

echo "== Priority 4: combined =="
if [[ "$ONSET_COUNT" -eq 0 ]]; then
    echo "ONSET/ONSETS not set - skipping combined." >&2
else
    for onset in "${ONSET_ARR[@]}"; do
        cond="combined"
        [[ "$ONSET_COUNT" -gt 1 ]] && cond="combined_onset${onset}"
        for seed in $SEEDS; do
            run_condition "$cond" "$seed" --mode combined --threshold "$TAU" \
                --onset "$onset" --keep_rate "$KEEP_RATE" --final_revision_epochs "$FINAL_REVISION_EPOCHS"
        done
    done
fi

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Done. ${run_count} run(s) printed (dry-run), ${skip_count} skipped (already completed)."
else
    echo "Done. ${run_count} run(s) attempted, ${skip_count} skipped (already completed), ${#FAILED_CONDITIONS[@]} failed."
    if [[ "${#FAILED_CONDITIONS[@]}" -gt 0 ]]; then
        echo "Failed (each retried ${MAX_RETRIES}x before giving up):"
        for f in "${FAILED_CONDITIONS[@]}"; do
            echo "  - $f"
        done
        exit 1
    fi
fi
