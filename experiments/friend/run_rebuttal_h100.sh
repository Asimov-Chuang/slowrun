#!/usr/bin/env bash
# =============================================================================
# Slowrun ICLR rebuttal: extra tuned runs + alignment checks against PSC + warmup ablation,
# on 4 H100s, no cluster scheduler. Instructions: experiments/friend/README.md
#
#   GPUS=0,1,2,3 bash experiments/friend/run_rebuttal_h100.sh all
#   GPUS=... bash experiments/friend/run_rebuttal_h100.sh <stage>   # setup|data|probe|run|summary|pack
#   GPUS=... DRY_RUN=1 bash experiments/friend/run_rebuttal_h100.sh run   # print the job list
#
# Run from a clone of https://github.com/TongQM/slowrun (branch fix/lr-schedule-overrun), inside
# tmux/screen. Safe to stop and restart: finished jobs are skipped, interrupted ones resume from
# their last epoch checkpoint.
#
# Settings (environment variables):
#   GPUS        required. GPU indices we may use, e.g. 4,5,6,7
#   WORKDIR     venv, checkpoints, state of this script (default $HOME/slowrun_rebuttal)
#   FORCE=1     continue past a failed platform probe (only if the authors say so)
#   BUDGET_H=60 a warmup run starts only if it would finish within this many hours of the
#               start of the run stage; otherwise it is skipped
#
# Jobs. Tuned recipe = AdamW weight decay + 20% linear LR cooldown, 40 epochs, P = 100M tokens.
#   probe      3 x 8 min  step-1/50 training loss at L6/L12/L18 W768 vs PSC; stops on mismatch
#   A_*        2 runs     alignment: same seed + config as existing PSC runs
#                         L12/W768 member 0, lambda 0.2 (PSC 3.5380); L6/W768, lambda 0.15 (PSC 3.6224)
#   M_*        3 runs     ensemble member L12/W768, lambda 0.1 / 0.15 / 0.3
#   S_*        1 run      4x-deeper single L48/W768, lambda 0.1 (PSC swept 0.15 / 0.2 / 0.3)
#   E_*, R_*   0 or 3+1   if a member lambda beats 0.2: members 1-3 at it, then E = 2,3,4 ensembles
#   W_*        6 runs     after the above: paper recipe (lambda 0, constant LR) with a linear
#                         warmup of 0 / 300 / 1500 steps, L6/W768 20 epochs and L18/W768 14 epochs
# About 52-67 GPU-hours; ~20 h wall on 4 H100-SXM.
# =============================================================================
set -uo pipefail

STAGE="${1:-all}"
GPUS="${GPUS:-}"
WORKDIR="${WORKDIR:-$HOME/slowrun_rebuttal}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
BUDGET_H="${BUDGET_H:-60}"
MOCK="${MOCK:-0}"            # testing only: fake training, no GPU
POLL="${POLL:-60}"           # seconds between queue checks while waiting on dependencies

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
TRAIN_COMMIT="0848c2c18adacd0124be11f95203dcc658f3e106"      # training code must equal this commit
TORCH_VERSION="2.9.1"        # the PSC version; keep it
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

VENV="$WORKDIR/venv"
CKPT="$WORKDIR/checkpoints"
LOGS="$CODE/experiments/logs"
STATE="$WORKDIR/state"
TOOLS="$WORKDIR/tools"
QUEUE="$STATE/queue.txt"       # static job list
QUEUE2="$STATE/queue2.txt"     # jobs generated after the member sweep (scanned first)
LOCK="$STATE/queue.lock"
GROUP="rebuttal_h100"
PY="$VENV/bin/python"

[ -n "$GPUS" ] || { echo "Set GPUS to the GPU indices you may use, e.g. GPUS=4,5,6,7"; exit 1; }
mkdir -p "$WORKDIR" "$STATE" "$TOOLS"
exec > >(tee -a "$WORKDIR/run.log") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "ERROR: $*"; exit 1; }

# Everything that trains runs with the hub offline, so FlashAttention-3 can never be fetched:
# every PSC run used PyTorch SDPA with full causal attention, and the results must match that.
train_env() {
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=offline WANDB_SILENT=true
    export TIKTOKEN_CACHE_DIR="$WORKDIR/tiktoken_cache"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export PYTHONUNBUFFERED=1
}

# =============================================================================
# setup: check the code, Python 3.11 venv, torch 2.9.1 (cu128), no flash-attn
# =============================================================================
stage_setup() {
    say "== setup (repo $CODE)"
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv | tee "$STATE/gpus.csv"
    IFS=, read -ra G <<< "$GPUS"
    for g in "${G[@]}"; do
        cap=$(nvidia-smi -i "$g" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) || die "GPU $g not found"
        [ "${cap%%.*}" = 9 ] || say "WARNING: GPU $g has compute capability $cap (not H100)"
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "${used:-0}" -lt 1000 ] || say "WARNING: GPU $g already has ${used} MiB in use"
    done
    avail=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc 0-9)
    [ "${avail:-0}" -ge 60 ] || say "WARNING: only ${avail} GB free in $WORKDIR (60 GB recommended)"

    git -C "$CODE" cat-file -e "$TRAIN_COMMIT" 2>/dev/null || git -C "$CODE" fetch -q origin || true
    git -C "$CODE" diff --quiet "$TRAIN_COMMIT" -- unlimited/train.py experiments/parallel/replay_fused.py prepare_data.py \
        || die "training code differs from $TRAIN_COMMIT; check out branch fix/lr-schedule-overrun unmodified"
    say "code $(git -C "$CODE" log --oneline -1); training code identical to ${TRAIN_COMMIT:0:7}"

    if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
        python3 -m pip install --user -q uv 2>/dev/null \
            || { command -v pipx >/dev/null && pipx install uv; } \
            || curl -LsSf https://astral.sh/uv/install.sh | sh \
            || die "could not install uv; install it (https://docs.astral.sh/uv/) and rerun"
    fi
    UV=$(command -v uv || echo "$HOME/.local/bin/uv")
    # uv's managed Python ships its C headers, which triton needs to build kernels
    [ -x "$PY" ] || "$UV" venv --python 3.11 "$VENV" || die "uv venv failed"
    "$UV" pip install --python "$PY" -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX" || die "torch install failed"
    "$UV" pip install --python "$PY" -q numpy tiktoken datasets tqdm wandb || die "package install failed"
    # never install these: they would switch attention to FA3/FA2 with sliding windows
    "$UV" pip uninstall --python "$PY" -q kernels flash-attn flash_attn 2>/dev/null || true

    "$PY" - <<'EOF' || die "environment check failed"
import importlib.util, sysconfig, os, torch
assert torch.cuda.is_available(), "CUDA not available to torch"
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()
for m in ("kernels", "flash_attn"):
    assert importlib.util.find_spec(m) is None, f"{m} is installed; uninstall it"
inc = sysconfig.get_paths()["include"]
assert os.path.exists(os.path.join(inc, "Python.h")), f"Python.h missing in {inc} (triton needs it)"
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
EOF
    { git -C "$CODE" log -1 --format='commit %H %s'; "$PY" -c "import torch;print('torch',torch.__version__,torch.version.cuda)";
      nvidia-smi; "$UV" pip freeze --python "$PY"; uname -a; } > "$STATE/env.txt" 2>&1
    write_tools
    say "setup OK"
}

# =============================================================================
# data: FineWeb sample-10BT, 100M train / 10M val tokens, GPT-2 tokenizer (~30-60 min)
# =============================================================================
stage_data() {
    say "== data"
    export TIKTOKEN_CACHE_DIR="$WORKDIR/tiktoken_cache"
    cd "$CODE" || die "no code dir"
    if [ ! -f fineweb_data/fineweb_train.pt ] || [ ! -f fineweb_data/fineweb_val.pt ]; then
        "$PY" prepare_data.py || die "prepare_data.py failed (needs access to huggingface.co)"
    fi
    "$PY" "$TOOLS/fingerprint.py" fineweb_data | tee "$STATE/data_fingerprint.txt"
}

common_flags() {  # L W ensemble_type model_idx epochs
    local L=$1 W=$2
    echo "--n_layer=$L --n_head=$((W / 64)) --n_embd=$W --num-models=5 --single-model-idx=$4 --ensemble-type=$3" \
         "--num-epochs=$5 --num-epochs-model-0=$5 --optimizer=adamw --ensemble-mode=logit --data-fraction=1.0" \
         "--val-every-n-steps=152 --total-batch-size=131072 --compile-mode=inductor" \
         "--completep --mup-base-width=768 --mup-base-depth=12 --mup-base-head-dim=64 --no-ve-projs"
}

# =============================================================================
# probe: does this machine reproduce PSC? (same-seed PSC reruns agree to 1e-4 at step 1)
# =============================================================================
stage_probe() {
    say "== probe"
    train_env; cd "$CODE" || die "no code dir"; mkdir -p "$LOGS"
    IFS=, read -ra G <<< "$GPUS"
    local i=0 pids=() cells=("6 768" "12 768" "18 768")
    for c in "${cells[@]}"; do
        read -r L W <<< "$c"
        local name="${GROUP}_probe_d${L}_w${W}" rd="${GROUP}_probe_d${L}_w${W}"
        rm -rf "${CKPT:?}/$rd" "$LOGS/$name.out" "$LOGS/$name.err"; mkdir -p "$CKPT/$rd"
        ( CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} timeout 10m "$VENV/bin/torchrun" --standalone --nproc_per_node=1 -- unlimited/train.py \
            $(common_flags "$L" "$W" init 0 40) --no-warmdown --weight-decay=0 \
            --checkpoint-base="$CKPT" --keep-epoch-ckpts-every=1000 --resume="$rd" --run="$name" --wandb_group="$GROUP" \
            > "$LOGS/$name.out" 2> "$LOGS/$name.err"; rm -rf "${CKPT:?}/$rd" ) &
        pids+=($!); i=$((i + 1))
    done
    wait "${pids[@]}"
    "$PY" "$TOOLS/probe_check.py" "$LOGS" "$GROUP" | tee "$STATE/probe_result.txt"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != 0 ]; then
        if [ "$FORCE" = 1 ]; then say "probe failed; continuing because FORCE=1"
        else stage_pack; die "platform probe failed. Send the tarball above to the authors before running anything else."; fi
    fi
}

# =============================================================================
# job list.  train line:  NAME|DEPS|train|L W ET IDX EPOCHS WD NOWARMDOWN WARMUP RUNDIR CLEAN HOURS
#   ET init = data order shared across models (the paper's single models, seed of model 0)
#   ET init_shuffle = independent data order per member (the paper's ensembles)
#   CLEAN=1 deletes the checkpoints when the run finishes; members keep theirs for the replay.
#   HOURS = expected wall time on one H100-SXM (used for the warmup time budget).
#   Longest job first, so the 16 h L48 run does not finish last. Warmup runs wait for SELECT and
#   come after everything else, so they only use GPUs the tuned runs no longer need.
# =============================================================================
write_queue() {
    [ -s "$QUEUE" ] && return
    cat > "$QUEUE" <<EOF
S_d48_w768_wd0.1|-|train|48 768 init 0 40 0.1 0 0 S_d48_w768_wd0.1 1 16.5
A_member_wd0.2|-|train|12 768 init_shuffle 0 40 0.2 0 0 mem_wd0.2 0 5
M_wd0.1|-|train|12 768 init_shuffle 0 40 0.1 0 0 mem_wd0.1 0 5
M_wd0.15|-|train|12 768 init_shuffle 0 40 0.15 0 0 mem_wd0.15 0 5
M_wd0.3|-|train|12 768 init_shuffle 0 40 0.3 0 0 mem_wd0.3 0 5
SELECT|A_member_wd0.2,M_wd0.1,M_wd0.15,M_wd0.3|select|
A_d6_w768_wd0.15|-|train|6 768 init 0 40 0.15 0 0 A_d6_w768_wd0.15 1 3.1
W_d18_wu0|SELECT|train|18 768 init 0 14 0 1 0 W_d18_wu0 1 2.4
W_d18_wu300|SELECT|train|18 768 init 0 14 0 1 300 W_d18_wu300 1 2.4
W_d18_wu1500|SELECT|train|18 768 init 0 14 0 1 1500 W_d18_wu1500 1 2.4
W_d6_wu0|SELECT|train|6 768 init 0 20 0 1 0 W_d6_wu0 1 1.6
W_d6_wu300|SELECT|train|6 768 init 0 20 0 1 300 W_d6_wu300 1 1.6
W_d6_wu1500|SELECT|train|6 768 init 0 20 0 1 1500 W_d6_wu1500 1 1.6
EOF
    touch "$QUEUE2"
}

# If a member lambda beats 0.2 (both trained here), train members 1-3 at it and replay E=2,3,4.
# Checkpoints of the members that are not needed are deleted.
do_select() {
    "$PY" "$TOOLS/select_members.py" "$LOGS" "$GROUP" > "$STATE/selection.txt" || return 1
    cat "$STATE/selection.txt"
    local best; best=$(grep -E '^BEST ' "$STATE/selection.txt" | awk '{print $2}')
    for wd in 0.1 0.15 0.2 0.3; do [ "$wd" != "$best" ] && rm -rf "${CKPT:?}/mem_wd$wd"; done
    if [ -z "$best" ] || [ "$best" = 0.2 ]; then rm -rf "${CKPT:?}/mem_wd0.2"; : > "$QUEUE2"; return 0; fi
    local tmp="$QUEUE2.tmp" deps="M_wd$best"; : > "$tmp"
    for idx in 1 2 3; do
        echo "E_wd${best}_m${idx}|-|train|12 768 init_shuffle $idx 40 $best 0 0 mem_wd$best 0 5" >> "$tmp"
        deps+=",E_wd${best}_m${idx}"
    done
    echo "R_wd${best}|${deps}|replay|mem_wd$best" >> "$tmp"
    mv "$tmp" "$QUEUE2"
}

# --------------------------------------------------------------------------- queue engine
jobs_all() { cat "$QUEUE2" "$QUEUE" 2>/dev/null | grep -v '^\s*$'; }
st() { for s in done fail skip run; do [ -e "$STATE/$1.$s" ] && { echo $s; return; }; done; echo pending; }

over_budget() {  # name hours -> true if a W_ job would end past BUDGET_H hours after the run start
    case $1 in W_*) ;; *) return 1 ;; esac
    local start now; start=$(cat "$STATE/run_start"); now=$(date +%s)
    awk -v s="$start" -v n="$now" -v h="$2" -v b="$BUDGET_H" 'BEGIN { exit !((n - s) / 3600 + h > b) }'
}

claim_next() {  # prints the claimed job line, or nothing
    (
        flock 9
        while IFS= read -r line; do
            local name=${line%%|*} rest=${line#*|}; local deps=${rest%%|*}
            [ "$(st "$name")" = pending ] || continue
            local ok=1
            if [ "$deps" != "-" ]; then
                IFS=, read -ra D <<< "$deps"
                for d in "${D[@]}"; do
                    case $(st "$d") in
                        done) ;;
                        fail|skip) echo "dependency $d did not complete" > "$STATE/$name.fail"; ok=0; break ;;
                        *) ok=0; break ;;
                    esac
                done
            fi
            [ "$ok" = 1 ] || continue
            local hours; hours=$(echo "$line" | cut -d'|' -f4 | awk '{print $NF}')
            if over_budget "$name" "$hours"; then
                echo "skipped: would end past ${BUDGET_H} h" > "$STATE/$name.skip"; continue
            fi
            echo "$$" > "$STATE/$name.run"; echo "$line"; break
        done < <(jobs_all)
    ) 9> "$LOCK"
}

all_finished() {
    local n=0
    while IFS= read -r line; do
        case $(st "${line%%|*}") in done|fail|skip) ;; *) n=$((n + 1)) ;; esac
    done < <(jobs_all)
    [ "$n" = 0 ]
}

run_train() {  # gpu name args...
    local gpu=$1 name=$2; shift 2
    local L=$1 W=$2 ET=$3 IDX=$4 EP=$5 WD=$6 NOWD=$7 WU=$8 RD=$9 CLEAN=${10}
    local out="$LOGS/${GROUP}_${name}.out" err="$LOGS/${GROUP}_${name}.err"
    local extra="--weight-decay=$WD"
    [ "$NOWD" = 1 ] && extra+=" --no-warmdown"
    [ "$WU" -gt 0 ] && extra+=" --warmup-steps=$WU"
    mkdir -p "$CKPT/$RD"
    if [ "$MOCK" = 1 ]; then "$PY" "$TOOLS/mock_train.py" "$name" "$EP" > "$out" 2> "$err"; sleep 1
    else
        for attempt in 1 2; do   # a crash resumes from the last epoch checkpoint once
            CUDA_VISIBLE_DEVICES=$gpu "$VENV/bin/torchrun" --standalone --nproc_per_node=1 -- unlimited/train.py \
                $(common_flags "$L" "$W" "$ET" "$IDX" "$EP") $extra \
                --checkpoint-base="$CKPT" --keep-epoch-ckpts-every=1000 \
                --resume="$RD" --run="${GROUP}_${name}" --wandb_group="$GROUP" >> "$out" 2>> "$err"
            grep -q "Total time" "$out" && break
            say "  $name exited without finishing (attempt $attempt)"
        done
    fi
    grep -q "Using PyTorch SDPA fallback" "$out" || { say "  $name: attention backend is not SDPA - results unusable"; return 3; }
    grep -q "Total time" "$out" || return 1
    [ "$CLEAN" = 1 ] && rm -rf "${CKPT:?}/$RD"
    return 0
}

run_replay() {  # gpu name rundir
    local gpu=$1 name=$2 rd=$3 out="$LOGS/${GROUP}_$2.out"
    if [ "$MOCK" = 1 ]; then printf '[epoch 40 ens=2] val_loss=3.47\n[epoch 40 ens=3] val_loss=3.44\n[epoch 40 ens=4] val_loss=3.43\n' > "$out"
    else
        CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/parallel/replay_fused.py --checkpoint-dir="$CKPT/$rd" \
            --num-models=4 --ens-sizes 2 3 4 --num-epochs=40 --start-epoch=40 --end-epoch=40 --eval-mode=epoch \
            --wandb-group="$GROUP" --wandb-run-name-prefix="${GROUP}_$name" > "$out" 2> "$LOGS/${GROUP}_$name.err" || return 1
    fi
    grep -q "ens=4\] val_loss=" "$out" || return 1
    rm -rf "${CKPT:?}/$rd"     # members are no longer needed
}

worker() {  # gpu
    local gpu=$1 line name type args rc
    while true; do
        line=$(claim_next)
        if [ -z "$line" ]; then all_finished && return 0; sleep "$POLL"; continue; fi
        name=${line%%|*}; type=$(echo "$line" | cut -d'|' -f3); args=$(echo "$line" | cut -d'|' -f4)
        say "GPU $gpu start  $name"
        case $type in
            train)  run_train "$gpu" "$name" $args; rc=$? ;;
            replay) run_replay "$gpu" "$name" "$args"; rc=$? ;;
            select) ( flock 9; do_select ) 9> "$LOCK.sel"; rc=$? ;;
        esac
        rm -f "$STATE/$name.run"
        if [ "$rc" = 0 ]; then touch "$STATE/$name.done"; say "GPU $gpu done   $name"
        else echo "exit $rc" > "$STATE/$name.fail"; say "GPU $gpu FAILED $name (exit $rc; see $LOGS/${GROUP}_$name.err)"; fi
        "$PY" "$TOOLS/summarize.py" "$LOGS" "$GROUP" "$STATE" > "$STATE/summary.tsv" 2>/dev/null || true
    done
}

stage_run() {
    write_queue
    if [ "$DRY_RUN" = 1 ]; then jobs_all | column -t -s'|'; return; fi
    say "== run on GPUs $GPUS (warmup budget ${BUDGET_H} h from the first start of this stage)"
    train_env; cd "$CODE" || die "no code dir"; mkdir -p "$LOGS"
    [ -s "$STATE/run_start" ] || date +%s > "$STATE/run_start"   # kept across restarts
    rm -f "$STATE"/*.run                       # left over from an interrupted session
    for f in "$STATE"/*.fail; do [ -e "$f" ] && grep -q "^exit 1$" "$f" && rm -f "$f"; done   # retry plain crashes
    IFS=, read -ra G <<< "$GPUS"
    local pids=()
    for g in "${G[@]}"; do worker "$g" & pids+=($!); sleep 5; done
    wait "${pids[@]}"
    stage_summary
}

stage_summary() {
    "$PY" "$TOOLS/summarize.py" "$LOGS" "$GROUP" "$STATE" | tee "$STATE/summary.tsv"
}

stage_pack() {
    local tb="$WORKDIR/slowrun_rebuttal_results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz"
    tar czf "$tb" -C "$WORKDIR" state run.log -C "$CODE/experiments" \
        $(cd "$CODE/experiments" && ls -d logs/${GROUP}_* 2>/dev/null) 2>/dev/null
    say "results: $tb ($(du -h "$tb" | cut -f1)) - send this file to the authors"
}

# =============================================================================
# helper scripts (written into $WORKDIR/tools)
# =============================================================================
write_tools() {
cat > "$TOOLS/fingerprint.py" <<'EOF'
"""Token-content fingerprint of the prepared data (independent of torch.save byte layout)."""
import hashlib, os, sys, torch
REF = {"fineweb_val.pt": "a44f9f864cee0415", "fineweb_train.pt": "07edaf52b54d9563"}  # TACC copy
for f in ("fineweb_val.pt", "fineweb_train.pt"):
    p = os.path.join(sys.argv[1], f)
    d = torch.load(p, weights_only=False)
    h = hashlib.sha256()
    for c in d["chunks"]:
        h.update(c.to(torch.int32).numpy().tobytes())
    fh = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"{f}: chunks={len(d['chunks'])} seqs={sum(d['valid_counts'])} content={h.hexdigest()[:16]} "
          f"(TACC {REF[f]}, {'same' if h.hexdigest()[:16] == REF[f] else 'DIFFERENT'}) file_sha256={fh}")
EOF
cat > "$TOOLS/probe_check.py" <<'EOF'
"""Compare step-1/50 training loss with PSC (H100-SXM, torch 2.9.1, SDPA). Exit 1 on mismatch."""
import re, sys
logs, group = sys.argv[1], sys.argv[2]
PSC = {(6, 768): (10.9632, 7.2567), (12, 768): (10.9704, 7.1340), (18, 768): (10.9684, 7.3030)}
TOL1 = 0.002          # PSC reruns agree to 1e-4; the TACC A100 differed by 0.009-0.012
bad = 0
for (L, W), (r1, r50) in PSC.items():
    t = open(f"{logs}/{group}_probe_d{L}_w{W}.out", errors="ignore").read()
    sdpa = "Using PyTorch SDPA fallback" in t
    s1 = re.search(r"step 1/\d+ \| loss: ([0-9.]+)", t)
    s50 = re.search(r"step 50/\d+ \| loss: ([0-9.]+)", t)
    tps = re.findall(r"step \d+/\d+ \| loss: [0-9.]+ \| (\d+) tok/s", t)
    if not (s1 and s50):
        print(f"d{L}_w{W}: no step-1/50 loss in log (crash? see .err)"); bad = 1; continue
    a1, a50 = float(s1[1]), float(s50[1])
    ok = sdpa and abs(a1 - r1) <= TOL1
    bad |= not ok
    print(f"d{L}_w{W}: step1 {a1:.4f} (PSC {r1:.4f}, d={a1 - r1:+.4f})  step50 {a50:.4f} (PSC {r50:.4f}, "
          f"d={a50 - r50:+.4f})  sdpa={sdpa}  tok/s={tps[-1] if tps else '?'}  {'OK' if ok else 'MISMATCH'}")
print("PROBE", "FAILED" if bad else "PASSED")
sys.exit(1 if bad else 0)
EOF
cat > "$TOOLS/select_members.py" <<'EOF'
"""Rank the member weight decays by minimum validation loss (all trained on this machine)."""
import re, sys
logs, group = sys.argv[1], sys.argv[2]
RUNS = {"0.2": "A_member_wd0.2", "0.1": "M_wd0.1", "0.15": "M_wd0.15", "0.3": "M_wd0.3"}
res = []
for wd, run in RUNS.items():
    t = open(f"{logs}/{group}_{run}.out", errors="ignore").read()
    v = [float(x) for x in re.findall(r"val @ step \d+\] val_loss=([0-9.]+)", t)]
    if v:
        res.append((min(v), wd))
res.sort()
for v, wd in res:
    print(f"# member L12/W768 lambda {wd}: min val {v:.4f}")
if res:
    print("BEST", res[0][1])
EOF
cat > "$TOOLS/summarize.py" <<'EOF'
"""One row per run: min val loss and where, final val loss, plus PSC reference values."""
import glob, os, re, sys
logs, group, state = sys.argv[1:4]
PSC = {  # PSC H100 results for the same configs and seeds
    "A_member_wd0.2": "PSC same seed: 3.5380 (members 1-3: 3.5500/3.5482/3.5513; E2/E3/E4 3.4664/3.4431/3.4324)",
    "A_d6_w768_wd0.15": "PSC same seed: 3.6224",
    "M_wd0.1": "PSC lambda 0.2 member 0: 3.5380", "M_wd0.15": "PSC lambda 0.2 member 0: 3.5380",
    "M_wd0.3": "PSC lambda 0.2 member 0: 3.5380",
    "S_d48_w768_wd0.1": "PSC L48/W768 lambda 0.15/0.2/0.3: 3.4962/3.5185/3.5092",
    "W_d6_wu0": "PSC same seed: min 3.9682 @ step 9764",
    "W_d18_wu0": "PSC same seed: min 3.8894 @ step 5949",
}
def status(name):
    for s in ("done", "fail", "skip"):
        if os.path.exists(f"{state}/{name}.{s}"):
            return s
    return "running"
print("run\tstatus\tstep1\tmin_val\tmin_step\tfinal_val\tlast_step\ttok_per_s\treference")
for f in sorted(glob.glob(f"{logs}/{group}_*.out")):
    name = os.path.basename(f)[len(group) + 1:-4]
    if name.startswith("probe"):
        continue
    t = open(f, errors="ignore").read()
    ens = re.findall(r"\[epoch (\d+) ens=(\d+)\] val_loss=([0-9.]+)", t)
    if ens:
        print(f"{name}\t{status(name)}\t\t" + " ".join(f"E{e}={v}" for _, e, v in ens) + "\t\t\t\t\t"); continue
    v = [(int(s), float(x)) for s, x in re.findall(r"val @ step (\d+)\] val_loss=([0-9.]+)", t)]
    s1 = re.search(r"step 1/\d+ \| loss: ([0-9.]+)", t)
    tps = re.findall(r"\| (\d+) tok/s", t)
    ref = PSC.get(name, "")
    if v:
        m = min(v, key=lambda x: x[1])
        print(f"{name}\t{status(name)}\t{s1[1] if s1 else ''}\t{m[1]:.4f}\t{m[0]}\t{v[-1][1]:.4f}\t{v[-1][0]}\t{tps[-1] if tps else ''}\t{ref}")
    else:
        print(f"{name}\t{status(name)}\t{s1[1] if s1 else ''}\t\t\t\t\t\t{ref}")
for f in sorted(glob.glob(f"{state}/*.skip")):
    print(f"{os.path.basename(f)[:-5]}\tskip\t\t\t\t\t\t\t{open(f).read().strip()}")
EOF
cat > "$TOOLS/mock_train.py" <<'EOF'
import random, sys
name, ep = sys.argv[1], int(sys.argv[2])
print("Using PyTorch SDPA fallback (no FA3/FA2 available; sliding window attention disabled)")
print("  [epoch 1] [model 1] step 1/763 | loss: 10.9704 | 5905 tok/s")
for s in range(152, 763 * ep, 152 * 20):
    print(f"    [model 1 val @ step {s}] val_loss={3.5 + random.random() * 0.1:.4f}")
print("Total time: 0h 0m 1.0s")
EOF
}

# =============================================================================
case "$STAGE" in
    setup)   stage_setup ;;
    data)    stage_data ;;
    probe)   stage_probe ;;
    run)     [ -f "$TOOLS/summarize.py" ] || write_tools; stage_run ;;
    summary) stage_summary ;;
    pack)    stage_pack ;;
    all)     stage_setup && stage_data && stage_probe && stage_run && stage_pack ;;
    *)       die "unknown stage $STAGE (setup|data|probe|run|summary|pack|all)" ;;
esac
