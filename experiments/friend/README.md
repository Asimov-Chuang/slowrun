# Rebuttal experiments on 4 H100s

Instructions for running the slowrun rebuttal experiments on a machine outside our cluster.
They are written so a person or a coding agent can follow them. Everything is in one script,
[`run_rebuttal_h100.sh`](run_rebuttal_h100.sh). Follow the steps in order and do not change the
code or any hyperparameter: the point of these runs is to be directly comparable with runs
we already have.

## What this runs and why

The paper compares an ensemble of four 85M-parameter models against a single model with four
times the parameters, at matched compute. These jobs add the missing hyperparameter tuning for
that comparison and check that this machine reproduces our earlier numbers.

| Jobs | What | Purpose |
|---|---|---|
| `probe` (3 × 8 min) | first 50 training steps of three models | Machine check: step-1 loss must match our cluster's to within 0.002 |
| `A_member_wd0.2`, `A_d6_w768_wd0.15` (5 h, 3 h) | exact reruns of two existing runs | Alignment: final loss should match our values (3.5380, 3.6224) to within ~0.005 |
| `M_wd0.1`, `M_wd0.15`, `M_wd0.3` (5 h each) | ensemble member L12/W768 at three weight decays | Tune the ensemble side |
| `S_d48_w768_wd0.1` (16.5 h) | single model L48/W768 at weight decay 0.1 | Tune the single-model side |
| `E_*`, `R_*` (3 × 5 h + 20 min, only if needed) | three more members and the 2/3/4-member ensemble | Created automatically if a member weight decay beats 0.2 |
| `W_*` (6 runs, 1.6–2.4 h) | warmup of 0 / 300 / 1500 steps | A separate reviewer question; runs last, only if it fits in 60 h |

Total: about 52–67 GPU-hours, roughly 20 hours of wall time on 4 GPUs. Results are small text logs.
Checkpoints are deleted as soon as they are no longer needed. Peak disk use is about 30 GB.

## Requirements

- Linux, 4 NVIDIA H100 GPUs (80 GB) that nobody else is using during the run, driver for CUDA 12.x or newer.
- `git`, `python3`, `tmux` (or `screen`), about 60 GB of free disk.
- Internet access during setup and data preparation, to `github.com`, `pypi.org`,
  `download.pytorch.org` and `huggingface.co`. Training itself runs offline.

## Steps

1. **Get the code** (the branch, not `main`):

   ```bash
   git clone -b fix/lr-schedule-overrun https://github.com/TongQM/slowrun.git
   cd slowrun
   ```

2. **Pick the GPUs.** Use only the GPU indices the machine's owner assigned, for example `4,5,6,7`.
   Check that they are idle with `nvidia-smi`.

3. **Start everything in tmux**, so it survives a disconnect:

   ```bash
   tmux new -s slowrun
   GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh all
   ```

   `all` runs the stages in order:
   - **setup**: creates a Python 3.11 environment with PyTorch 2.9.1 in `~/slowrun_rebuttal/venv`.
   - **data**: downloads and tokenizes 110M FineWeb tokens, which takes 30–60 minutes.
   - **probe**: the machine check.
   - **run**: all the jobs.
   - **pack**: the results tarball.

   Detach with `Ctrl-b d`, and reattach with `tmux attach -t slowrun`.

4. **If the probe fails**, the script stops by itself and writes a results tarball. Send that
   tarball to the authors and wait. Do not rerun with `FORCE=1` unless the authors ask for it.

5. **Monitor**, without interrupting the run:

   ```bash
   tail -f ~/slowrun_rebuttal/run.log                         # job start / finish lines
   column -t -s $'\t' ~/slowrun_rebuttal/state/summary.tsv     # per-run losses so far
   nvidia-smi                                                 # one training process per assigned GPU
   ```

   Each job writes `experiments/logs/rebuttal_h100_<job>.out`. A healthy run prints
   `Using PyTorch SDPA fallback` near the top, then training steps with a `tok/s` rate.
   The rate should be about 270k tok/s for L12/W768 and about 80k tok/s for L48/W768.

6. **When it finishes**, the last line of `run.log` gives the path of
   `~/slowrun_rebuttal/slowrun_rebuttal_results_<host>_<date>.tar.gz`. Send that one file to the
   authors. It contains only logs and tables, a few MB.

## If something goes wrong

- **Interrupted** (reboot, killed tmux, crash). Rerun the run stage, then pack:

  ```bash
  GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh run
  GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh pack
  ```

  Finished jobs are skipped, and interrupted ones continue from their last epoch. The 60-hour
  warmup budget counts from the first start, so it is not reset by a restart.
- **A job is marked FAILED in `run.log`.** Its `.err` file in `experiments/logs/` has the error.
  The script already retries a crashed run once. Do not edit the training code. Send the
  `.err` file, or run `pack` and send the tarball, to the authors.
- **A log says `Using Flash Attention 3` or `Flash Attention 2`.** The run is not comparable
  with ours and the script marks it failed. Make sure no `flash-attn` or `kernels` package is
  installed in `~/slowrun_rebuttal/venv`.
- **Setup fails installing `uv`.** Install it as described at https://docs.astral.sh/uv/, then
  rerun the same command.
- **Data preparation cannot reach `huggingface.co`** (or is rate-limited). Retry later, or set
  `HF_TOKEN` to a Hugging Face read token before running the data stage.
- **The GPUs are needed back.** Stop with `Ctrl-c` in the tmux window. All finished results
  are kept. Run `pack` to send what exists so far.

## Do not

- Change `GPUS` to GPUs that were not assigned to us.
- Edit `unlimited/train.py`, the hyperparameters in the script, or the package versions.
- Install `flash-attn` or `kernels` into the environment.
- Send checkpoints. The tarball is all we need.
