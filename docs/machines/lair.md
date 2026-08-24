# LAIR (IU Luddy)

Machine-specific notes for setting up and running compression benchmarks on the Luddy
AI Research (LAIR) cluster. These notes summarize the LAIR Confluence documentation last
updated June 17, 2026. If this file and the current LAIR documentation disagree, follow
the LAIR documentation or contact `sicehelp@iu.edu`.

## Access

- Login host: `lair.luddy.indiana.edu`
- Authentication: IU username and passphrase over SSH
- Network: connect to the IU VPN before SSH when off campus
- Operating system: Ubuntu 22.04
- Scheduler: Slurm
- Compute hardware: L40S and H100 GPU nodes; the login node has no GPU

Example SSH configuration:

```sshconfig
Host lair
    HostName lair.luddy.indiana.edu
    User <IU-username>
```

On the affected Windows OpenSSH 9.5 client, the default cipher/MAC combination failed
against LAIR with `Corrupted MAC on input`, although the same connection worked from
WSL OpenSSH 9.6. The following host-local workaround avoids that combination:

```sshconfig
Host lair
    HostName lair.luddy.indiana.edu
    User <IU-username>
    Ciphers aes256-gcm@openssh.com
```

Keep this override scoped to LAIR rather than placing it under `Host *`.

VS Code Remote-SSH is not explicitly discussed in the LAIR documentation. Its
authenticated SSH tunnel and lightweight server process appear consistent with the
documented access rules, but this is an inference rather than an explicit approval.
Keep builds, tests, benchmarks, and heavy language-server/indexing work off the login
node. Ask `sicehelp@iu.edu` if explicit confirmation is needed.

## Resource-use rules

- Use the login node for lightweight editing, repository management, and job submission.
- Run processes requiring significant CPU time or memory as Slurm jobs.
- Request GPU resources only when they will be actively used. Idle GPU allocations may
  be terminated.
- Interactive allocations are for development and debugging and have a maximum wall
  time of 12 hours.
- Batch jobs have a maximum wall time of 14 days; the default is 1 hour when `--time`
  is omitted.
- Every job must specify a project Slurm account with `--account`.
- Specify either `H100` or `L40S` in GPU requests. Do not rely on a generic GPU request
  when collecting machine-specific benchmark results.

Find the Slurm accounts available to the current user:

```bash
sacctmgr show association format=account user="$USER"
```

Start a short interactive H100 session for building or smoke testing:

```bash
srun --account=<project> --gres=gpu:H100:1 --time=02:00:00 --pty bash
```

Use `gpu:L40S:1` instead when intentionally targeting an L40S. Leave `--gres` out of
CPU-only setup jobs.

Minimal batch header for an H100 benchmark:

```bash
#!/usr/bin/env bash
#SBATCH --account=<project>
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=benchkit
#SBATCH --output=benchkit-%j.out
#SBATCH --error=benchkit-%j.err
```

`general` is the documented default partition. Confirm the active partition and QoS
configuration with `scontrol show partition` before creating LAIR-specific submission
scripts.

## Storage

| Path | Intended use | Protection and limits |
|---|---|---|
| `/u/$USER` | Source, configuration, virtual environments, lightweight interactive files | 100 GB default quota; daily backup |
| `/data/user/$USER` | Private datasets, builds, and results not ready to share | 1 TB default quota; snapshots but no off-site backup |
| `/data/project/<project>` | Shared project datasets, builds, and results | Project quota; snapshots but no off-site backup |
| `/scratch/local/$USER` | Node-local temporary benchmark working files | Not shared between nodes; files older than 60 days are deleted |

Recommended initial layout:

```text
/u/$USER/research/compression_benchmarking/   repository and Python environment
/data/user/$USER/compression/                 persistent private data
/data/user/$USER/compression/sdrbench_data/   benchmark corpus
/data/user/$USER/compression/results/         retained benchmark results
/scratch/local/$USER/benchkit-work/           per-job disposable work files
```

Do not put I/O-intensive batch workloads in the home directory. Node-local scratch is
fast but is different on every compute node, so a job must stage input into it and copy
required results out before exiting.

Check quotas and capacity with:

```bash
xfs_quota -c 'quota -h' /u
df -h "/data/user/$USER"
df -h /data/project/<project>
```

For small transfers, LAIR permits `scp` and `rsync`. Globus Connect Personal is the
recommended option for larger transfers and is available as an environment module.

## Software setup checklist

Perform discovery before writing `scripts/env-lair.sh`; LAIR's documentation does not
list exact CUDA, compiler, CMake, or Python module names.

```bash
module avail
module spider cuda 2>/dev/null || true
module spider python 2>/dev/null || true
module spider cmake 2>/dev/null || true
which gcc g++ cmake ninja python3 nvcc
gcc --version
cmake --version
python3 --version
nvcc --version
nvidia-smi
```

Run `nvidia-smi` and GPU-dependent checks inside a Slurm allocation, not on the login
node. Record the selected module versions and paths in the eventual environment script
so batch jobs reproduce the interactive build environment.

LAIR provides Apptainer as a module and does not provide Docker. Load it with:

```bash
module load apptainer
```

Use `apptainer --nv` for NVIDIA GPU access and bind `/data` explicitly when a container
needs the user or project data filesystems. LAIR recommends Ubuntu 22.04-based container
images to match the host libc.

## Toolkit configuration to add after library setup

Once the compressor builds and dataset location are known:

1. Copy `configs/site.example.yaml` to the gitignored `configs/site.local.yaml`.
2. Set `fzgmod_cli` to the intended LAIR build, not an unrelated binary on `PATH`.
3. Set `results_root` under `/data/user/$USER` or the project data directory.
4. Add `scripts/env-lair.sh` containing the verified module loads, virtual environment,
   dataset root, results root, and compressor binary paths.
5. Add a LAIR Slurm submission script with an explicit account and GPU type.
6. Capture `nvidia-smi`, compiler, CUDA, driver, hostname, and Slurm job provenance with
   every baseline run.

Example local site configuration:

```yaml
fzgmod_cli: /u/<user>/research/FZGPUModules/build/release/bin/fzgmod-cli
results_root: /data/user/<user>/compression/results
```

Verify the actual FZGPUModules binary path after building; the example is a proposed
layout, not a claim about an existing LAIR installation.

## Network restrictions

Authenticated SSH local forwarding is the approved way to reach an application running
on an internal compute node. Services that expose LAIR to the public internet or bypass
user authentication, including Cloudflare Tunnels and ngrok, are prohibited. Contact
LAIR support before introducing any other externally reachable service.
