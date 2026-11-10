# Running the rebuttal experiments

One script, [`run_rebuttal_h100.sh`](run_rebuttal_h100.sh), runs everything on 4 H100 GPUs:
setup, data, a machine check, all training jobs, and packing the results. It takes about 20 hours.
Do not edit the code or the hyperparameters.

## 1. Check the machine

```bash
nvidia-smi             # 4 idle H100 80GB GPUs assigned to you; note their indices
df -h ~                # at least 60 GB free
which git python3 tmux
```

Setup needs internet access to github.com, pypi.org, download.pytorch.org and huggingface.co.

## 2. Get the code

```bash
git clone -b fix/lr-schedule-overrun https://github.com/TongQM/slowrun.git
cd slowrun
```

## 3. Start the run inside tmux

Replace `4,5,6,7` with the indices of your assigned GPUs.

```bash
tmux new -s slowrun
GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh all
```

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t slowrun`.

The script runs these stages in order and prints a timestamped line for each:

| Stage | Time | Done when `~/slowrun_rebuttal/run.log` shows |
|---|---|---|
| setup | ~10 min | `setup OK` |
| data | 30–60 min | two lines starting `fineweb_val.pt:` / `fineweb_train.pt:` |
| probe | ~10 min | `PROBE PASSED` |
| run | ~20 h | `GPU <n> start` / `done` for each job, then the summary table |
| pack | seconds | `results: .../slowrun_rebuttal_results_<host>_<date>.tar.gz` |

If the probe prints `PROBE FAILED`, the script stops and writes the results tarball. Send it to us
(step 6) and wait. Do not rerun with `FORCE=1` unless we ask.

## 4. Check progress

```bash
tail -n 30 ~/slowrun_rebuttal/run.log
column -t -s $'\t' ~/slowrun_rebuttal/state/summary.tsv
nvidia-smi
```

- `run.log`: which jobs have started, finished, or failed.
- `summary.tsv`: the validation losses so far.
- `nvidia-smi`: should show one python process on each assigned GPU.

Each job logs to `experiments/logs/rebuttal_h100_<job>.out`. A healthy log has
`Using PyTorch SDPA fallback` near the top, followed by training lines with a `tok/s` rate:

| Model | Expected rate |
|---|---|
| `L12/W768` | ~270k tok/s |
| `L48/W768` | ~80k tok/s |

## 5. If it stops or a job fails

- **Interrupted** (reboot, tmux closed, Ctrl-c). Run these two commands. Finished jobs are
  skipped and unfinished ones resume from their last epoch.

  ```bash
  cd slowrun
  GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh run
  GPUS=4,5,6,7 bash experiments/friend/run_rebuttal_h100.sh pack
  ```

- **`FAILED <job>` in `run.log`.** The error is in `experiments/logs/rebuttal_h100_<job>.err`.
  Crashed jobs are retried once automatically, and again on the next `run`. If it keeps failing,
  run `pack` and send us the tarball.
- **Setup or data stage failed.** Fix the reported problem, then rerun the same `all` command.
  Completed steps are reused.
  - `uv` will not install: install it from https://docs.astral.sh/uv/.
  - huggingface.co is rate-limited: wait, or `export HF_TOKEN=<read token>`.

## 6. Send the results

Send us the one file named on the last line of `run.log`:

```
~/slowrun_rebuttal/slowrun_rebuttal_results_<host>_<date>.tar.gz
```

It holds logs and tables only, a few MB. Checkpoints are deleted automatically.
`rm -rf ~/slowrun_rebuttal` frees the remaining disk space afterwards. Do this only once we
confirm we received the file.
