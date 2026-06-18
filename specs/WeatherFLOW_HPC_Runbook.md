# WeatherFLOW — UCT HPC Runbook

**User:** jrvpra001 · **Server:** hpc.uct.ac.za (= hex.uct.ac.za) · **Password:** your UCT password
**GPU grant:** `l40sugrd` group, `l40s` partition, `l40s` cards, **24 h max per job**
**CPU billing account:** `compsci` · **Job limits:** 120 cores total, 40 cores/job, 1 node/job
**Storage:** `/home/jrvpra001` 10 GB (backed up) · `/scratch/jrvpra001` 50 GB (volatile, NOT backed up)

---

## 0. The three lines that encode your grant

GPU training jobs:
```bash
#SBATCH --partition=l40s
#SBATCH --account=l40sugrd
#SBATCH --gres=gpu:l40s:1        # lowercase L; one card per job (CUDA limit)
#SBATCH --time=24:00:00          # your hard cap
```
CPU-only jobs (preprocessing, comparison, plots):
```bash
#SBATCH --partition=ada          # a normal CPU partition
#SBATCH --account=compsci
#SBATCH --time=08:00:00
```

---

## 1. First-time setup

**1.1 Connect** (run the UCT VPN first if off-campus):
```bash
ssh jrvpra001@hpc.uct.ac.za
quota                 # check /home and /scratch usage
ls -l                 # you have example-script.sh and gpu-example-script.sh here — read them
cat gpu-example-script.sh   # your real --account value is embedded in this file
```
Never run training/preprocessing on the head node — it will get your account suspended. Use `sintx` (interactive) or `sbatch` (batch).

**1.2 Lay out scratch** (everything heavy lives here):
```bash
mkdir -p /scratch/jrvpra001/weatherflow/{logs,cache,outputs}
cd /scratch/jrvpra001/weatherflow
```

**1.3 Get the code** (submodules matter — the 8 model repos are submodules):
```bash
git clone --recurse-submodules git@github.com:praise-jaravani/WeatherFLOW.git
# if you used HTTPS keys aren't set up: clone via https and `git submodule update --init --recursive`
```
Code can live in `/home` (small) or `/scratch`. If in `/home`, keep `cache/` and `outputs/` pointed at `/scratch`.

**1.4 Get the container there** (it's ~11 GB — it will NOT fit in 10 GB /home):
```bash
# from your local machine (WSL), push the built .sif up:
rsync -avP weatherflow.sif jrvpra001@hpc.uct.ac.za:/scratch/jrvpra001/weatherflow/
```
> Build the `.sif` locally or on a fakeroot-capable host — you can't build a container as root on the cluster. Make sure the image's PyTorch/CUDA supports **`sm_89` (L40S / Ada)**, not just `sm_80` (A100). An NGC PyTorch base image covers both.

**1.5 Get ERA5 onto scratch.** The CDS download is network-bound. Compute nodes may not have outbound internet — test first:
```bash
sintx --account=compsci --partition=ada --ntasks=2 --time=02:00:00
curl -sI https://cds.climate.copernicus.eu | head -1   # 200/301 => internet works here
```
- If internet works in the interactive session: run `python scripts/download_era5.py` there (resumable).
- If it doesn't: download locally and `rsync` the cache up, or run the (light, network-bound) download on the head node briefly — but that's borderline against the no-head-node-work policy, so prefer the interactive session or local-then-rsync route.
Point the downloader/cache at `/scratch/jrvpra001/weatherflow/cache`.

---

## 2. Smoke test — interactive, before any batch job

Confirms the container sees the GPU and the pipeline runs end to end at toy scale.
```bash
sintx --account=l40sugrd --partition=l40s --gres=gpu:l40s:1 --ntasks=4 --time=01:00:00

# once the prompt changes to a worker node:
SIF=/scratch/jrvpra001/weatherflow/weatherflow.sif
cd /scratch/jrvpra001/weatherflow/WeatherFLOW

singularity exec --nv "$SIF" nvidia-smi            # must list an L40S
singularity exec --nv "$SIF" python -m cli train \
    --model agcrn --dataset era5 --task regression --runs 1 --max-nodes 10
exit
```
`--nv` is mandatory — it binds the NVIDIA driver into the container. Without it the GPU is invisible.
> If `singularity` isn't on PATH: `module avail singularity` (or `apptainer`) then `module load ...`.

This is also where you run the **post-fix verification** from the update spec: confirm `mae_extreme["t2m"]` is non-zero and `units == "physical"` in the smoke summary.

---

## 3. Real training run — batch script

Save as `scripts/jobs/train_gpu.sh`:
```bash
#!/bin/sh
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4                       # CPU cores for the dataloader; keep modest
#SBATCH --time=24:00:00
#SBATCH --job-name="wf_mtgode_era5_reg"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/slurm-%j.out

SIF=/scratch/jrvpra001/weatherflow/weatherflow.sif
cd /scratch/jrvpra001/weatherflow/WeatherFLOW

singularity exec --nv \
    --bind /scratch/jrvpra001:/scratch/jrvpra001 \
    "$SIF" \
    python -m cli train --model mtgode --dataset era5 --task regression --runs 10
```
Submit and watch:
```bash
sbatch scripts/jobs/train_gpu.sh
squeue -u jrvpra001          # or: qstat   (your jobs only)
tail -f /scratch/jrvpra001/weatherflow/logs/slurm-<JOBID>.out
scancel <JOBID>              # to kill
```
Visual monitor: the dashboard at https://hpc.uct.ac.za/db (GPU utilisation, updates each minute).

---

## 4. The experimental matrix under the 24 h cap

Your scope: 4 models (agcrn, clcrn, mtgode, climode) × 2 datasets (era5, saws) × 2 tasks (regression, classification) × 10 seeds. The 24 h ceiling forces chunking.

**First, the timing run decides the unit of work.** From the spec's P1.11 — run MTGODE and ClimODE once at full 999 nodes for a few epochs and measure per-epoch wall-clock. Then:
- If **10 seeds of one (model, dataset, task) fit in 24 h** → one job per combination (16 jobs for your 4 models). Simplest.
- If not → **job array over seeds**, one seed per array task:

`scripts/jobs/train_array.sh`:
```bash
#!/bin/sh
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --job-name="wf_climode_era5_reg"
#SBATCH --mail-user=JRVPRA001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --array=42-51%2                   # seeds 42..51; %2 = at most 2 running at once
#SBATCH --output=/scratch/jrvpra001/weatherflow/logs/%x-%A_%a.out

SIF=/scratch/jrvpra001/weatherflow/weatherflow.sif
cd /scratch/jrvpra001/weatherflow/WeatherFLOW

singularity exec --nv --bind /scratch/jrvpra001:/scratch/jrvpra001 "$SIF" \
    python -m cli train --model climode --dataset era5 --task regression \
                        --runs 1 --seed ${SLURM_ARRAY_TASK_ID}
```
- `%2` throttles concurrency so you stay within the free-pool limits and don't hog cards.
- Each task writes to its own per-seed output dir; aggregate afterwards with a CPU job.
- ClimODE on L40S: if fp64 tolerances make it slow, this 12 h-per-seed split is your safety margin.

**Aggregation / stats / plots run as a cheap CPU job** (`--account=compsci --partition=ada`), calling `compare_results.py --wilcoxon --plot` once all seeds are in.

---

## 5. Data hygiene — `/scratch` is volatile

After each completed run, pull results back to a backed-up or off-cluster location:
```bash
# copy outputs to backed-up /home (watch the 10 GB quota — summaries/plots only, not raw checkpoints)
rsync -avP /scratch/jrvpra001/weatherflow/outputs/<run_dir>/summary.json \
           /home/jrvpra001/weatherflow_results/

# or pull to your local machine
rsync -avP jrvpra001@hpc.uct.ac.za:/scratch/jrvpra001/weatherflow/outputs/ ./results/
```
Keep an eye on `quota` — 10-seed runs with checkpoints will fill 50 GB fast. Store only what you need (summaries, per-seed metrics, final checkpoints); prune intermediate checkpoints.

---

## 6. Pre-flight checklist

- [ ] VPN up (if off-campus); `ssh jrvpra001@hpc.uct.ac.za` works.
- [ ] `.sif` is in `/scratch` and supports `sm_89` (L40S).
- [ ] `singularity exec --nv "$SIF" nvidia-smi` lists an L40S inside the container.
- [ ] ERA5 fully downloaded into `/scratch/.../cache`; cache rebuilt post-download (derive fix).
- [ ] GPU jobs use `--account=l40sugrd --partition=l40s --gres=gpu:l40s:1`, ≤ 24 h.
- [ ] CPU jobs use `--account=compsci`.
- [ ] One GPU per job (never `gpu:l40s:2`).
- [ ] Launch with `sbatch`, never `bash`; nothing heavy on the head node.
- [ ] Timing run done → matrix chunking decided (per-combination vs per-seed array).
- [ ] Results rsync'd off `/scratch` after each run.
- [ ] Acknowledge UCT HPC in the paper (required by their Acceptable Use Policy).

---

### Quick reference

| Task | Command |
|---|---|
| Login | `ssh jrvpra001@hpc.uct.ac.za` |
| Interactive GPU | `sintx --account=l40sugrd --partition=l40s --gres=gpu:l40s:1 --ntasks=4 --time=01:00:00` |
| Submit | `sbatch scripts/jobs/train_gpu.sh` |
| My queue | `squeue -u jrvpra001` (or `qstat`) |
| Cancel | `scancel <JOBID>` |
| Run in container | `singularity exec --nv "$SIF" python -m cli train ...` |
| Disk usage | `quota` |
| Dashboard | https://hpc.uct.ac.za/db |
