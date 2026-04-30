# nite-eval on k3s

Deploy nite-eval as a self-running CronJob on a single-node k3s homelab. The
container layout mirrors the bare-metal `scripts/run_nightly.sh` flow: judges
stay up as long-running Deployments on the Tesla P40, the target llama-swap
runs on the RTX 3090, and the orchestrator fires nightly as a Kubernetes Job.

> **TL;DR**
> 1. Build + push three images to GHCR.
> 2. Fill in `secret.yaml` with your GPU UUIDs and apply it.
> 3. `kubectl apply -k k8s/base/` brings up everything except the manual Job.
> 4. CronJob fires at `06:00 UTC` (23:00 PT). Reports land in the `nite-eval-results` PVC.

---

## Architecture

```
              ┌─ Tesla P40 ─────────────────────────────┐
              │  judge-flow Deployment   :9092          │
              │  judge-reward Deployment :9091          │
              └─────────────────────────────────────────┘
                         ▲
                         │ HTTP (in-cluster Services)
                         │
┌─ Orchestrator (CronJob) ┼──────────────► PVC ──┐
│  python -m nite_eval     │   /workspace/        │
│  no GPU                  │   results/           │
└──────────────────────────┘                      │
                         │                         ▼
                         │ HTTP                SQLite DB
                         ▼                     + reports
              ┌─ RTX 3090 ──────────────────────────────┐
              │  target-llama-swap Deployment :9070     │
              │   ↳ swaps qwen3.5-9b / 27b / gemma4 …   │
              └─────────────────────────────────────────┘
```

Service DNS the orchestrator uses (in-cluster):

| Service | URL |
|---|---|
| target llama-swap | `http://target-llama-swap.nite-eval.svc.cluster.local:9070` |
| Flow-Judge | `http://judge-flow.nite-eval.svc.cluster.local:9092/v1` |
| RewardAnything | `http://judge-reward.nite-eval.svc.cluster.local:9091/v1` |

---

## Prerequisites

- k3s with the NVIDIA container runtime registered (containerd v2 config, `runtimeClassName: nvidia`).
- `local-path` storage class (k3s default).
- GGUF model files on the node:
  - Target models (qwen3.5/3.6, gemma4) → `/home/woojay/models/` (override via deployment patch if your path differs).
  - Judge models (`Flow-Judge-v0.1.Q6_K.gguf`, `RewardAnything-8B-v1.Q6_K.gguf`) → same dir.
- GHCR access for image push (`gh auth login` or a `GITHUB_TOKEN` PAT with `write:packages`).

---

## 1. Build images

From the repo root:

```bash
# Orchestrator (Python; no GPU)
docker build -f Dockerfile.orchestrator -t ghcr.io/w00jay/nite-eval-orchestrator:latest .

# Judge (llama-server, both CUDA archs 6.1 + 8.6)
docker build -f Dockerfile.judge        -t ghcr.io/w00jay/nite-eval-judge:latest .

# Target (llama-swap + llama-server, both archs)
docker build -f Dockerfile.target       -t ghcr.io/w00jay/nite-eval-target:latest .
```

Tips:
- The judge and target builds compile llama.cpp from source. Expect ~10–15 min on first build, ~1 min cached.
- Override the llama.cpp ref or arch list at build time:
  ```bash
  docker build -f Dockerfile.judge \
    --build-arg LLAMACPP_REF=master \
    --build-arg CUDA_ARCHITECTURES="86" \
    -t ghcr.io/w00jay/nite-eval-judge:3090-only .
  ```

## 2. Push to GHCR

```bash
echo "$CR_PAT" | docker login ghcr.io -u w00jay --password-stdin
docker push ghcr.io/w00jay/nite-eval-orchestrator:latest
docker push ghcr.io/w00jay/nite-eval-judge:latest
docker push ghcr.io/w00jay/nite-eval-target:latest
```

The `nite-eval` namespace pulls public images, no `imagePullSecrets` needed.

## 3. Configure GPU UUIDs

```bash
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-aaaa....)
# GPU 1: Tesla P40 (UUID: GPU-bbbb....)

cp k8s/base/secret.yaml.example k8s/base/secret.yaml
$EDITOR k8s/base/secret.yaml   # paste in TARGET_GPU_UUID / JUDGE_GPU_UUID

kubectl apply -f k8s/base/secret.yaml
```

`secret.yaml` is intentionally **not** in `kustomization.yaml`, and is gitignored.
Apply it out-of-band so a populated copy never lands in version control.

## 4. Apply manifests

```bash
kubectl apply -k k8s/base/
```

This brings up:
- namespace `nite-eval`
- ConfigMaps (`eval-config`, `llama-swap-config`)
- PVC (`nite-eval-results`, 5Gi `local-path`)
- Deployments + Services (`judge-flow`, `judge-reward`, `target-llama-swap`)
- The `nite-eval-orchestrator` CronJob (schedule `0 6 * * *` UTC)

The manual-trigger Job (`job-orchestrator.yaml`) is **excluded** from
kustomize so a bulk apply doesn't fire an ad-hoc run. Apply it when you need it.

## 5. Verify the stack is up

```bash
# Pods
kubectl -n nite-eval get pods,svc,pvc

# Judges should reach Ready in ~1–3 min once the GGUF mmaps in.
kubectl -n nite-eval logs deploy/judge-flow      -f
kubectl -n nite-eval logs deploy/judge-reward    -f
kubectl -n nite-eval logs deploy/target-llama-swap -f

# Hit a judge directly to confirm:
kubectl -n nite-eval port-forward svc/judge-flow 9092:9092 &
curl -s http://127.0.0.1:9092/health
```

## 6. Trigger a run

### Manual (no schedule wait)

Two equivalent options:

```bash
# Option A: apply the named manual Job manifest
kubectl -n nite-eval delete job nite-eval-orchestrator-manual --ignore-not-found
kubectl apply -f k8s/base/job-orchestrator.yaml

# Option B: spawn from the live CronJob's PodTemplate
kubectl -n nite-eval create job adhoc-$(date +%Y%m%d-%H%M) \
  --from=cronjob/nite-eval-orchestrator
```

### Watch the run

```bash
kubectl -n nite-eval get jobs --watch
kubectl -n nite-eval logs job/nite-eval-orchestrator-manual -f
```

### Read results

The PVC holds SQLite + Markdown reports. To pull them out:

```bash
# Snag a shell into the running orchestrator (or any pod that mounts the PVC)
kubectl -n nite-eval exec -it job/nite-eval-orchestrator-manual -- bash
ls /workspace/results/runs/

# Copy a specific report to the local machine:
kubectl -n nite-eval cp \
  nite-eval-orchestrator-manual-xxxxx:/workspace/results/runs/run-20260429-060000.md \
  ./run-20260429-060000.md
```

For a longer-lived option, run a small `busybox` debug pod with the PVC mounted.

## 7. Verify checkpoint resume

The orchestrator writes per-(model, task) checkpoints to SQLite as it runs.
Killing the pod mid-run and recreating it should pick up where it left off.

```bash
# 1. Start a manual run.
kubectl apply -f k8s/base/job-orchestrator.yaml

# 2. Wait until you see the first model start scoring tasks.
kubectl -n nite-eval logs job/nite-eval-orchestrator-manual -f

# 3. Kill the pod (the Job's restartPolicy=Never means no auto-respawn).
kubectl -n nite-eval delete pod -l app.kubernetes.io/name=nite-eval-orchestrator,nite-eval/trigger=manual

# 4. Recreate the Job (a clean Job re-uses the same SQLite via the PVC).
kubectl -n nite-eval delete job nite-eval-orchestrator-manual
kubectl apply -f k8s/base/job-orchestrator.yaml

# 5. The new run resumes from the last checkpointed (model, task) — confirm
#    in logs that completed pairs are skipped before any new tasks run.
```

The PVC persists across pod restarts, so the SQLite results DB is the resume
point. To force a clean run, delete the run row in SQLite or use a fresh
`run_id` (orchestrator generates one per invocation).

---

## Resource notes

| Pod | CPU req/limit | Memory req/limit | GPU |
|---|---|---|---|
| `judge-flow` | 2 / 4 | 8Gi / 12Gi | P40 (UUID-pinned) |
| `judge-reward` | 2 / 4 | 10Gi / 14Gi | P40 (UUID-pinned) |
| `target-llama-swap` | 4 / 8 | 24Gi / 32Gi | RTX 3090 (UUID-pinned) |
| `orchestrator` (Job) | 1 / 2 | 1Gi / 2Gi | — |

GPU pinning intentionally uses `NVIDIA_VISIBLE_DEVICES=GPU-...` (Secret-sourced)
instead of `nvidia.com/gpu: 1` resource requests. The NVIDIA k8s device plugin
and direct UUID exposure conflict on mixed-arch hosts (3090 + P40), and the
bare-metal pipeline already proves UUID pinning works. If you ever migrate to
a uniform-GPU node, switching back to `nvidia.com/gpu` resource requests is a
one-line change in each Deployment.

---

## CronJob schedule

Default `0 6 * * *` UTC = 23:00 PT during PDT, 22:00 PT during PST.
Edit `cronjob-orchestrator.yaml` (`.spec.schedule` and `.spec.timeZone`) to change.
Set `spec.suspend: true` to pause without deleting the resource.

`terminationGracePeriodSeconds: 3600` gives the orchestrator up to one hour
after SIGTERM to checkpoint the in-flight task. `activeDeadlineSeconds: 36000`
caps a single run at 10 hours.

---

## Common failure modes

### Pod stays Pending — "0/1 nodes available"

Confirm the node has the runtime class registered:
```bash
kubectl get runtimeclass nvidia
```
If missing, register it (one-time):
```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

### Judge Pod runs but llama-server can't see the GPU

Double-check the Secret value is the full UUID with `GPU-` prefix:
```bash
kubectl -n nite-eval get secret nite-eval-gpu-pinning -o jsonpath='{.data.JUDGE_GPU_UUID}' | base64 -d
```
And confirm the runtime sees it:
```bash
kubectl -n nite-eval exec deploy/judge-flow -- nvidia-smi
```
If `nvidia-smi` errors or shows zero GPUs, the runtime class isn't being
honored — verify `runtimeClassName: nvidia` is set on the Pod and the
containerd config has the `nvidia` runtime registered.

### Judge Pod OOM (CrashLoopBackOff with exit code 137)

Bump `resources.limits.memory` on the affected Deployment. RewardAnything
(8B Q6_K) is the typical offender; 14Gi is a tight ceiling for ctx 4096.

### Target llama-swap "model file not found"

The hostPath in `target-llama-swap-deployment.yaml` is `/home/woojay/models`.
Either:
- patch the path with kustomize, or
- symlink your GGUF dir into that location.

Confirm the filenames in `configmap-llama-swap.yaml` match the GGUFs on disk
(they're a starting point, not a guarantee):
```bash
kubectl -n nite-eval exec deploy/target-llama-swap -- ls /models
```

### CronJob fires but PVC is full

```bash
kubectl -n nite-eval exec deploy/target-llama-swap -- df -h /workspace
```
Bump `pvc-results.yaml`'s `storage` request and re-apply, or prune old
`results/runs/` directories.

### Image pull fails on first apply

GHCR images are public, but your cluster's containerd may need a one-time
DNS/connectivity check:
```bash
kubectl -n nite-eval describe pod <pending-pod> | tail -30
```
If you see `ImagePullBackOff`, confirm the cluster can reach `ghcr.io`.

---

## Layout

```
k8s/
├── README.md                          # this file
└── base/
    ├── kustomization.yaml             # kustomize entrypoint
    ├── namespace.yaml
    ├── configmap-eval.yaml            # eval_config.yaml (in-cluster URLs)
    ├── configmap-llama-swap.yaml      # llama_swap_config template
    ├── secret.yaml.example            # GPU UUID template (apply secret.yaml separately)
    ├── pvc-results.yaml               # 5Gi local-path PVC
    ├── judge-flow-deployment.yaml
    ├── judge-flow-service.yaml
    ├── judge-reward-deployment.yaml
    ├── judge-reward-service.yaml
    ├── target-llama-swap-deployment.yaml
    ├── target-llama-swap-service.yaml
    ├── cronjob-orchestrator.yaml      # nightly schedule
    └── job-orchestrator.yaml          # manual trigger (excluded from kustomize)
```

---

## What this doesn't do (yet)

These are deliberate omissions for v1; track them as v2 work:

- **No Prometheus / metrics export.** SQLite + Markdown reports are the durable record.
- **No Slack / email notifications** on success or failure.
- **No multi-node support.** PVC is `ReadWriteOnce` and Deployments use hostPath for GGUFs.
- **No vLLM / Ollama backend** — llama.cpp + llama-swap only.
- **No GitOps.** Plain `kubectl apply -k`.
- **No Ingress / cluster-external access.** All Services are ClusterIP.
- **No automatic image rebuild** on llama.cpp upstream changes.

For the bare-metal fallback path, see [`scripts/run_nightly.sh`](../scripts/run_nightly.sh).
