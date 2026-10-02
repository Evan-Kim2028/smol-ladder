# AMD Developer Cloud runbook: three SFT arms on an MI300X, end to end

Research and plan for the owner's $100 AMD Developer Cloud credit. Code in [`ops/amd/`](../ops/amd/),
tests in [`tests/test_ops_amd.py`](../tests/test_ops_amd.py). Nothing in this repository has been
run on AMD hardware; every number below is either quoted from a source or labelled an estimate.

The task: run three LoRA SFT arms of `Qwen/Qwen3.5-2B` on one MI300X — **A** (upstream's
`FineEnvs/SmolDataEnvs-sft`), **B** (our exported `ja3` trajectories), **A+B** (both) — evaluate each
with the ladder harness on the SmolDataEnvs `test` split, plus the base control, and keep every
artifact.

**The headline finding, before anything else.** `docs/PLAN.md` assumes $1.99/GPU-hour and ~50 GPU
hours. DigitalOcean's own AMD pricing page says **$2.59/hour**, so the credit buys **~38.6 GPU
hours, not ~50**. Everything below is costed at $2.59. The $1.99 figure appears in
[third-party coverage](https://lilting.ch/en/articles/amd-developer-cloud-credit-journey) and
could not be confirmed on any AMD or DigitalOcean page. Check the console price before applying the
credit; if it really is $1.99 there, raise `AMD_PRICE_PER_GPU_HOUR` and the plan gets more room.

---

## 1. How AMD Developer Cloud actually works

### 1a. Who runs it, and that decides everything else

**AMD Developer Cloud is DigitalOcean infrastructure, sold by AMD.** AMD's own product page:

> "AMD provides developers easy access to AMD Instinct™ MI300X GPUs **through a third-party cloud
> website**." ([amd.com](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html))

> "Pay-As-You-Go — **Digital Ocean** — Get instant access to AMD Instinct GPUs via Digital Ocean."
> ([same page](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html))

> "How can I get support? **The AMD Developer Cloud is hosted by a third-party cloud provider.**
> Contact your cloud provider for support [here](https://cloudsupport.digitalocean.com/)."

This is the single most consequential fact in this document, and it is good news: because the
droplets are ordinary DigitalOcean droplets, **there is a real, documented API and CLI** (§1c).
An AMD-specific walled garden would have meant no programmatic lifecycle at all.

Console: **https://devcloud.amd.com** (AMD SSO; it redirects through `amd.digitalocean.com`). Sign
in with "AMD Sign-in" — a [third-party account's write-up](https://lilting.ch/en/articles/amd-developer-cloud-credit-journey)
warns that following the DigitalOcean activation email lands on a login page with no identity
provider, and that this is a dead end rather than a bug to work around.

### 1b. Instances, images, and what they cost

| | |
|---|---|
| plan slug, 1 GPU | `gpu-mi300x1-192gb` — 192 GB VRAM, 20 vCPU, 240 GB RAM, 720 GB boot + 5 TB scratch |
| plan slug, 8 GPU | `gpu-mi300x8-1536gb` — 1536 GB VRAM, 160 vCPU, 1920 GB RAM, 1920 GB boot + 40 TB scratch |
| **price** | **$2.59/h (1 GPU), $20.72/h (8 GPU)** — [docs.digitalocean.com](https://docs.digitalocean.com/products/amd/details/pricing/) |
| granularity | "billed per second with a minimum charge of 60 seconds or $0.01, whichever is higher" |
| region | **ATL1 only** — [availability](https://docs.digitalocean.com/products/amd/details/availability/) |
| quota | "You can provision up to eight AMD MI300X GPU Droplets or one 8x AMD MI300X GPU Droplet using an AMD Developer Cloud credit." "**You cannot resize GPU Droplets.**" — [limits](https://docs.digitalocean.com/products/amd/details/limits/) |

**Use the 1-GPU plan.** The 8-GPU node is exactly 8× the price for a job that is single-GPU, and
AMD's own quota language charges quota by GPU-hours.

Images: AMD's [getting-started blog](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
describes a **Bare OS** option (Ubuntu variants, ROCm installable yourself) and **Quick Start**
ROCm-ready images: *Vanilla ROCm* (ROCm, no ML packages) or *Quick Start Packages* (a preloaded
Docker container — vLLM, SGLang, PyTorch, Megatron or JAX — entered with `docker exec -it rocm
bash`). DigitalOcean notes the AI/ML image is "based on Ubuntu 22.04".

> **UNKNOWN:** the exact image slugs/snapshots currently offered in the ADC droplet-creation screen.
> AMD's blog is from June 2025 and DigitalOcean's says "Ubuntu 22.04", while current ROCm images
> are Ubuntu 24.04. Read the actual list off the create page. `bootstrap.sh` deliberately does not
> hard-code an image slug: it detects what is there (`/dev/kfd`, `rocm-smi`/`amdsmi`) and the owner
> picks the ROCm-ready image when creating the droplet by hand.

SSH: log in as **`root@<public-IP>`** (AMD's blog). The key is attached at creation via the
"Add an SSH Key" button.

### 1c. API and CLI: yes, and this is the good news

DigitalOcean documents the ADC lifecycle through its **standard API and `doctl`**, in a page
written specifically for AMD GPU Droplets:

> "The single GPU Droplet size slug is `gpu-mi300x1-192gb`. The 8 GPU Droplet size slug is
> `gpu-mi300x8-1536gb`." — [how to create AMD GPU Droplets](https://docs.digitalocean.com/products/amd/how-to/create/)

The same page gives `doctl auth init` + `doctl compute droplet create`, and
`POST https://api.digitalocean.com/v2/droplets` with `Authorization: Bearer $DIGITALOCEAN_TOKEN`.
Standard endpoints cover the rest: `GET /v2/droplets` (list), `POST /v2/droplets/{id}/actions` with
`{"type":"power_off"}` ([actions reference](https://docs.digitalocean.com/reference/api/reference/droplet-actions/)),
and `DELETE /v2/droplets/{id}` / `doctl compute droplet delete`
([destroy](https://docs.digitalocean.com/products/droplets/how-to/destroy/)). There is **no**
`amdcloud`/`adcctl` and no AMD-hosted API.

> **UNKNOWN:** whether an ADC personal access token is scoped or restricted in any way beyond being
> a normal DigitalOcean PAT. DigitalOcean documenting `doctl` inside its ADC section strongly
> implies it behaves identically, but no page states that in those words. Confirm by creating a
> throwaway droplet with `read`-only first.

Token scope: **a DigitalOcean personal access token with `read` + `write`.** Read for
list/status, write for create/delete/power actions. There is no finer-grained scope that would let
us grant delete-only-but-not-create.

One constraint to know about: ADC resources live in a **separate team** from any regular
DigitalOcean account —

> "You cannot transfer AMD Developer Cloud resources to any of your DigitalOcean teams. Your AMD
> Cloud Developer teams are separate from any DigitalOcean teams you may have."

### 1d. Billing: the rule that decides the safety rails

**A powered-off GPU droplet still bills. Only destroying it stops the meter.** Two independent
official statements:

DigitalOcean:
> "Billing begins when you create the GPU Droplet and ends when you [destroy it]. **You are still
> billed for GPU Droplets that are powered off.** The GPU and other compute resources stay reserved
> on the hypervisor even when the Droplet is not running. Powering off a GPU Droplet does not stop
> billing. To end billing, [destroy the GPU Droplet]." ([pricing](https://docs.digitalocean.com/products/amd/details/pricing/))

AMD:
> "When you power off your GPU VM, you are still billed for it. This is because your disk space,
> CPU, RAM, and IP address are all reserved, even while powered off. Therefore, charges are made
> until you destroy the instance." ([AMD FAQ](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html))

This is why the watchdog **powers off and then tells you to destroy**, and why `driver.py --destroy`
exists separately from `--power-off`. Confusing those two is the single most expensive mistake
available here.

Storage and egress:

- Boot disk (720 GB) and scratch disk (5 TB) are **included** in the hourly rate. The scratch disk
  is local and non-persistent: "The scratch disk is not included in snapshots of the Droplet, and if
  you destroy or recreate a GPU Droplet, the scratch disk is lost."
- Block volumes are separate at **$0.10/GiB/month**, billed whether attached or not, and AMD's
  terms say complimentary credit "will not be offered" for volumes, object storage or backups.
  **Do not attach a volume; put nothing durable on a paid volume and expect the credit to cover it.**
- Egress: 15,000 GiB/month free for the 1-GPU plan, then **$0.01/GiB**; inbound is free. The
  allowance accrues with droplet lifetime, so a short-lived droplet earns proportionally little
  free egress — but our traffic is a few GB of model weights and adapters, so this is not a risk.

### 1e. The credit: how it is applied, and the card caveat

AMD:
> "AMD is offering an initial **$100** complimentary cloud credit to qualified developers who apply
> and join AMD AI Developer Program. Complimentary credit hours **expire thirty (30) days from the
> date of deposit** in your developer cloud account unless stated otherwise in your credit
> confirmation email. Credits are activated once cloud account is created or logged into via $100
> credit link provided." ([AMD FAQ](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html))

> "Upon application approval, complimentary credits will be deposited in your third-party AMD
> Developer Cloud account **within three (3) business days**. Credits will appear on the 'My AMD
> Home' page in the AMD Developer Cloud console." (same)

> "Cloud credit allocation on the AMD Developer Cloud is determined by AMD in its sole discretion,
> based on the intended use case (e.g., inference, training, finetuning), and a detailed
> description of how you plan to use the GPU credit."

DigitalOcean:
> "**You must provide a valid payment method to use the credit.** Once you have used the credit, your
> account is charged at the standard rates for the resources you provision."
> "You can only use these credits one time, per account, and they cannot be transferred or used to
> provision resources on existing DigitalOcean accounts."
> ([credits](https://docs.digitalocean.com/products/amd/details/credits/))

> "If your credit expires and you have no active payment method on file, your AMD GPU VM will be
> destroyed and you will lose access to the GPU(s) and the data on the GPU(s). **If payment method is
> added, it will automatically charge your payment method once credit is exhausted.**" (AMD FAQ)

So the mechanics are:

1. Join the AI Developer Program (separate from just having an AMD account — [the third-party
   account](https://lilting.ch/en/articles/amd-developer-cloud-credit-journey) documents being stuck
   at exactly this step).
2. Claim "Free Cloud Credit" under My Benefits, submit the application (AMD: 3 business days).
3. Activate the ADC account with AMD SSO; the credit appears under **Billing → Credits**, *not* the
   main balance, which shows $0.00 and looks like a failure. It is not.
4. **Add a payment method.** It is mandatory, and it is the thing that turns an overrun into a
   charge on a real card. The plan's `--budget` cap and the watchdog exist because of this sentence.

> **UNKNOWN (owner must check):** whether the credit has been applied to your account yet, its
> deposit date, and therefore when the 30-day clock expires. `docs/PLAN.md` records this as unknown
> and nothing in this repo can determine it. Read it off the console's Billing → Credits tab.

---

## 2. What the agent does, and what only the owner can do

### Mode 1 — owner creates, agent runs (recommended first)

**Credentials the owner needs: none beyond their existing AMD/DigitalOcean login.**
**Credentials the agent needs: an ssh key pair. No token, no API.**

| step | who | what |
|---|---|---|
| 1 | **owner** | Confirm the $100 is on the Credits tab; note the expiry date. |
| 2 | **owner** | Create the droplet in the console: plan `gpu-mi300x1-192gb`, region ATL1, the **ROCm-ready / AI-ML image** (not bare OS), and **attach the laptop's SSH public key** via "Add an SSH Key". |
| 3 | **owner** | Add a payment method (mandatory — §1e). |
| 4 | owner | Put `HF_TOKEN` and `AMD_HUB_NAMESPACE` in the repo's git-ignored `.env`; write `.env` to the droplet. |
| 5 | agent | `python ops/amd/driver.py --host <ip> --user root --identity ~/.ssh/id_ed25519` — bootstrap, smoke, three arms, four evaluations, sync, power off. |
| 6 | **owner** | **Destroy the droplet.** No token means the agent cannot do this, which is the point of this mode. |

This mode is the default because the destructive step stays with a human. It costs the owner one
console visit at the start and one at the end.

### Mode 2 — agent manages the lifecycle via API token

**Credential: one DigitalOcean personal access token with `read` + `write` scope.**

- Stored as `AMD_CLOUD_API_TOKEN` in the repo's **git-ignored `.env`** (`.gitignore` already lists
  `.env`). It is read by `driver.py` at startup, passed to `doctl` through the environment or
  `doctl auth init`, and never appears in an argv, a log line, or a commit. `driver.py --mode api`
  **refuses to run** without it and says where to put it.
- **Never commit it, never pass it as a flag.** A token in argv is in the shell history, in `ps`,
  and in this repo's test output.
- Revoke it in the DigitalOcean console when the work is done; it is the one credential here that
  can delete things.

With it the agent can create the droplet (`--create`), list/status it (`--status`), power it off
(`--power-off`) and **destroy it (`--destroy`)** — which is what makes unattended teardown possible
and removes the human from the loop entirely. Both modes' commands are printable offline with
`--dry-run`.

**Recommendation:** do Mode 1 first, because it is the one that cannot spend money by accident.
Add the token only when the whole path has been walked once by hand.

---

## 3. The run plan, timed and costed

### The estimates, and what they are derived from

**Every GPU-hour below is an estimate. None is a measurement.** The one measured number in this
repository is the laptop smoke run in `docs/TRAINING.md`: a **0.8B** QLoRA at 2048 tokens,
**30 steps in 22 min** on a power-limited RTX 4050. From it:

- arm A is **4,439 rows** (from `docs/TRAINING.md` §4) at `--max-length 8192`, batch 1 × accum 8
  ⇒ ~555 optimizer steps for one epoch.
- The laptop cannot be scaled to MI300X arithmetically in any way that would survive scrutiny — a
  2B at 4× the context is a different workload, and bf16 LoRA removes the 4-bit overhead. So
  instead of a fake scaling factor, these are **order-of-magnitude figures with a wide error bar**,
  sized so the *budget* is safe rather than so the *estimate* is impressive.
- **Derivation, stated plainly:** one epoch of ~555 steps on a data-center GPU at bf16 LoRA,
  8192-token packing, for a 2B model — expected in the **1.5–3 h** range for arm A. Arm B (2,029
  rows per `ja3_sft.manifest.json`, ~45% of A) scales to ~1 h. A+B is the sum plus overhead,
  ~3 h. Evaluation is 0.75 h per model: a server start (~5 min) plus a 250-task ladder sweep, which
  `docs/LOCAL_MODELS.md` puts at ~25 min for `--agent program` and ~1–1.5 h for `--agent bash` on
  a 6 GB laptop at 150–250 tok/s aggregate; a 192 GB MI300X is far faster, but bash's tool-call
  overhead is protocol-bound rather than GPU-bound, so 0.75 h is a middle estimate.
- **Treat as ±2×, and re-derive after the first hour.** The smoke run and arm A's actual runtime
  are measured on the droplet; the runbook's job is to be *spending-side* safe, not accurate.

### The plan, in order

Wall-clock is cumulative from droplet creation. Ordering rationale is in the table's last column.

| # | step | clock | GPU-h | $ | why here |
|---|---|---|---|---|---|
| 1 | `bootstrap.sh --with-data --with-eval` | 0:00–0:45 | 0.75 | **1.94** | clone at a pinned commit, venv inside the ROCm container, HF login, rsync the SFT sets and task tables |
| 2 | start `watchdog.sh` | 0:45 | — | 0.00 | armed **before** anything that can hang |
| 3 | **smoke**: 20 SFT steps on arm A | 0:45–1:00 | 0.25 | **0.65** | proves bf16 LoRA trains on ROCm |
| 4 | **smoke**: serve base, 5 ladder tasks | 1:00–1:15 | 0.25 | **0.65** | proves vLLM serves and the grader runs on ROCm |
| 5 | **arm A** SFT | 1:15–4:15 | 2.50 | **6.47** | most rows + the published comparison; its measured runtime corrects the estimates for the rest |
| 6 | push A to the Hub | | | 0.00 | a killed run's only surviving copy |
| 7 | **arm A** eval (`bash`, 250 tasks × 5 rungs) | 4:15–5:15 | 0.75 | **1.94** | evaluate right after training, while the droplet is warm |
| 8 | rsync A's results back | | | 0.00 | the measurement is the result, not the adapter |
| 9–11 | **arm B** SFT → push → eval | 5:15–7:30 | 1.00 + 0.75 | **2.59 + 1.94** | |
| 12–14 | **arm A+B** SFT → push → eval | 7:30–11:30 | 3.00 + 0.75 | **7.77 + 1.94** | |
| 15–16 | **base control** eval (`program`) + results | 11:30–12:30 | 0.75 | **1.94** | needs no training, so it goes last; it is arm 0, the floor |
| 17 | final sync (Hub + rsync) | 12:30–12:45 | 0.25 | **0.65** | everything off the droplet |
| 18 | power off, then **destroy** | 12:45 | — | 0.00 | **only destroy stops billing** |
| | **TOTAL** | **~13 h** | **11.00** | **$28.5** | of a $100 credit / ~38.6 GPU-h |

`driver.py` computes this table, so the numbers here and the numbers it prints cannot drift:
`budget_report()` is the single definition, and a test parses this table's TOTAL row and asserts it
equals the tool's total. Run `python ops/amd/driver.py --host <ip> --dry-run` to print it and then
**every command in order**, executing none of them. That is the artifact to read before spending
anything.

(The tool's `budget_report` prints setup + the three SFT arms + four evaluations = **$27.8**; the
$28.5 above additionally counts the final sync's quarter hour. Both round to "about $28", ~28% of
the credit, leaving ~$71 — enough for a second seed on A and B (`docs/PLAN.md` asks for ≥2 seeds), a
`--samples 2` sweep, or the GRPO arms the plan still wants. `driver.py` warns when a plan is over
budget and when it is using less than half the cap, so the headroom is visible rather than assumed.)

---

## 4. Safety rails

### The budget cap

`AMD_BUDGET_USD` (default **80**, deliberately under the $100 credit) becomes
`AMD_WALLCLOCK_LIMIT_MIN` = `budget / price × 60`. At the default that is **1853 minutes ≈ 30.9 h**.
The watchdog fires at that wall clock, so the run stops *with credit left* rather than with a card
charge. Override the price if the console disagrees: `AMD_PRICE_PER_GPU_HOUR=1.99`.

**The cap is enforced by elapsed time, not by reading a balance**, because the only balance API is
the console and the droplet has no idea what it has spent. Time is a proxy, and it is exact enough
here: at a fixed hourly rate, elapsed hours × rate is spent dollars.

### The dead-man switch

`watchdog.sh`, started in the background at step 2 and looping on a `sleep` tick. **Three
independent trip conditions, any one sufficient:**

| condition | default | why |
|---|---|---|
| wall clock | `AMD_WALLCLOCK_LIMIT_MIN` (1853) | the budget ceiling |
| idle | `AMD_IDLE_LIMIT_MIN` (45 min) | no `sft_lora` / `run_ladder` / `vllm serve` process for 45 min |
| heartbeat | stale > 45 min | the laptop stopped touching `.heartbeat` |

The **heartbeat** is the one that catches the failure nobody sees: the laptop sleeps, the ssh
connection drops, the sweep keeps running, and nothing tells anyone. Every driver step `touch`es
`/opt/smol-ladder/.heartbeat`.

On a trip it attempts `sync_back.sh --push-hub`, powers the droplet off, and prints where to destroy
it. It **cannot destroy the droplet from inside** — a droplet cannot delete itself — which is why
`driver.py --destroy` (API mode) exists and why the runbook lists destroy as an owner step in Mode 1.

### Nothing is lost if the instance dies

- **Adapters → Hugging Face Hub, private repos.** `sft_lora.py --hub-model-id` makes TRL push on
  **every save** (`hub_strategy="every_save"`), so a kill costs at most one save interval and a
  fresh droplet can `--resume` from the Hub rather than from a wiped disk.
- **Logs and results → the same, plus rsync.** `push_artifacts.py` uploads adapters, logs and the
  results tree for this driver's run tags to a private **dataset** repo; `sync_back.sh --to-laptop`
  rsyncs the same tree back with `--partial` (resumable) and, deliberately, **no `--delete`** — this
  droplet's tree is a strict subset of the laptop's shared results tree, and `--delete` here would
  remove every other agent's runs.
- **Order matters.** Hub first, laptop second: the Hub copy is the one that survives if the droplet
  dies mid-rsync.

### What happens to billing in each failure mode

| failure | what is lost | billing | recovery |
|---|---|---|---|
| laptop sleeps / disconnects mid-run | nothing (heartbeat trips first) | **keeps billing** until destroyed | watchdog powers off at 45 min; destroy to stop |
| a training step hangs or NaNs | that arm's partial work | keeps billing | `--resume` from the last checkpoint (on the Hub) |
| the droplet itself dies / is lost | nothing if Hub push was on | stops (nothing to bill) | recreate, `bootstrap.sh`, `--resume` |
| ssh key rejected / cannot log in | nothing yet | **keeps billing from creation** | destroy in console, re-create with the right key |
| driver run finished normally | nothing | **still billing** | destroy — this is step 18 |
| credit exhausted with no card | **the droplet and all data on it** | ends | nothing; this is why the Hub push is not optional |
| credit exhausted **with** a card | nothing | **charges the card** | the budget cap is the only thing preventing this |

---

## 5. The first-hour ROCm smoke test

Run **all** of this before arm A. Each line is a failure this stack actually has, not a formality.
`driver.py` runs steps 3–4 as `smoke-sft` and `smoke-eval`; the rest is manual and is what the
first hour is for.

1. **The GPU is visible.** `rocm-smi --showproductname` or `amdsmi static` — **not** `nvidia-smi`,
   which does not exist here. Expect MI300X, 192 GB. If `/dev/kfd` is missing, the image has no
   driver and nothing below will work.
2. **torch sees it, in bf16.**
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
   — on ROCm `torch.cuda` is the correct namespace; there is no separate `torch.rocm`.
3. **A matmul is right.** `torch.randn(4096,4096, dtype=bf16) @ itself` vs the float32 result:
   `allclose(atol=1e-2)`. MI300X is CDNA 3 — bf16 is native, fp16 is emulated and slower, and
   `fp32` has no TF32-style fast path.
4. **The stack imports on ROCm.** `import transformers, peft, trl, accelerate, datasets` and print
   versions. These are pure Python over torch, which is the reason the plan expects them to work.
5. **A LoRA step actually reduces loss.** `run_sft.sh --arm A --smoke --inside-rocm`: 20 steps,
   loss must fall. If it NaNs, the dtype is wrong (`--precision bf16`).
6. **vLLM boots and answers.** `run_eval.sh --arm base --limit 5 --rungs L1`. Watch for three
   specific failures, in this order of likelihood:
   - `Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now` → the vLLM
     build is too old (`docs/LOCAL_MODELS.md` says ≥0.16.2, and this is the most likely smoke
     failure of all).
   - ROCm-specific: vLLM v0.30 **removed the `CUDA_VISIBLE_DEVICES` fallback on ROCm** — use
     `HIP_VISIBLE_DEVICES`, which `serve.sh` already does.
   - The ladder prints scores rather than errors. A sweep that completes with a 0.00 pass rate is a
     *plausible but wrong* result: check that tool calls are actually being parsed before believing
     a low number.
7. **Tool calling returns a real `tool_call`.** Under `--agent bash`, `--enable-auto-tool-choice
   --tool-call-parser qwen3_coder` must produce a parsed call, not an empty one. `docs/LOCAL_MODELS.md`
   flags this as untested for the 2B; it is the first thing to check if auto tool calls come back
   empty.
8. **The template is the non-thinking one.** `--default-chat-template-kwargs '{"enable_thinking":
   false}'`. Without it Qwen3.5-2B will not stop generating, and a run that trains one template and
   evaluates another is the template-mismatch failure every arm in this study is measured against.
9. **HF push works.** `huggingface-cli whoami`, then one small upload. A 13-hour run that fails to
   push at hour 12 is the failure this check exists to prevent.
10. **The watchdog is armed.** `cat /opt/smol-ladder/.watchdog-state`, confirm the cap, and
    `tail /var/log/smol-ladder/watchdog.log`. Run it with `--max-minutes 2 --idle-minutes 1` once
    to watch it fire.

---

## 6. ROCm and the stack: what is confirmed, what is not

### Containers and versions

The current official ROCm matrix ([compatibility matrix, ROCm 10.0.0](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html))
lists, for MI300X / CDNA 3 / gfx942: **PyTorch 2.13.0, 2.12.0, 2.11.0**; **vLLM 0.27.0**;
SGLang 0.5.15; on Python 3.14/3.13/3.12. Container tags, read live from Docker Hub:

- [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch): `rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`,
  and `rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` / `2.11.0` — note the several py3.11–3.14
  variants, so pick the tag whose Python matches what the rest of the stack wants.
- [`rocm/vllm`](https://hub.docker.com/r/rocm/vllm): `rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`,
  and the newest `rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`.

> **UNKNOWN:** whether the ADC droplet's own preloaded container matches any of these. The AMD blog's
> `docker exec -it rocm bash` container is not the `rocm/vllm` image and its contents are not
> documented. **`run_sft.sh --inside-rocm` is written to pull the repo's own container**
> (`AMD_ROCM_CONTAINER`, default `smol-rocm`) so the ROCm version is pinned by us rather than
> inherited — but building/pulling that image is a manual step (`bootstrap.sh` does not build it).

A **pip route also exists** and may avoid a container entirely — vLLM's v0.30.0 release notes list:
> "ROCm | `pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723`"
> and "ROCm | `docker pull vllm/vllm-openai-rocm:v0.30.0`"

That is a **newer vLLM (0.30.0) than the ROCm matrix's 0.27.0**, and `docs/LOCAL_MODELS.md` requires
≥0.16.2 for Qwen3.5 support — so 0.30.0 comfortably clears the bar, and is the version the plan's
`vllm==0.30.0` pin refers to. It is built against ROCm 7.23 (`rocm723`), so it must match the
droplet's driver.

### The rest of the stack

| | status | source |
|---|---|---|
| **transformers / peft / trl / accelerate / datasets** | Pure Python over torch. ROCm needs no special handling — torch is the only GPU-coupled layer, and it ships in the image. | `docs/transformers/en/installation` lists CUDA / CPU / XPU / Spark tabs only; there is no ROCm tab because there is nothing ROCm-specific to do. |
| **bitsandbytes (QLoRA)** | **Avoid, as `docs/PLAN.md` says.** ROCm support is a *preview alpha*: "At present, the Intel CPU and **AMD ROCm** backends are considered fully functional… we are currently in the alpha testing phase, **bugs are expected, and performance might not meet expectations**." ([non_cuda_backends](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends)) | Also unnecessary: 192 GB fits bf16 LoRA with room to spare, and `sft_lora.py` is run **without** `--load-in-4bit`. |
| **flash-attention** | Two separate things. (a) PyTorch's own SDPA flash backend on ROCm is native — "With the release of PyTorch 2.3 for ROCm, Flash Attention is now natively integrated into the `F.scaled_dot_product_attention` function." (b) The `flash_attn` *library* is a different story: "we cannot simply run `pip install flash-attn` because it installs a version that is not compatible with AMD GPUs. Instead, we need to clone AMD's flash-attention repo and build it from source" ([ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html), May 2024). | **Do not build it for this run.** Qwen3.5 uses GatedDeltaNet/linear attention, for which transformers falls back to unfused kernels — `docs/TRAINING.md` already records `causal_conv1d` and `flash-linear-attention` as absent-but-not-needed, and nothing in the SFT path asks for `attn_implementation="flash_attention_2"`. |
| **vLLM tool-call parser** | `--tool-call-parser qwen3_coder` with `--enable-auto-tool-choice`. | From the vLLM Qwen3.5 recipe via `docs/LOCAL_MODELS.md`; the combination is **untested** for a 2B dense model and is smoke-test step 7. |

### MI300X specifics that will bite

- **`HIP_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`.** vLLM v0.30 removed the CUDA-variable
  fallback on ROCm; setting the CUDA one is silently ignored. `serve.sh` uses the ROCm name.
- **No `nvidia-smi`.** `rocm-smi` or `amdsmi` (ROCm 10 ships AMD SMI 27.0.0). Any script that shells
  out to `nvidia-smi` to check memory is wrong here.
- **One 1-GPU plan, 192 GB.** No tensor parallelism, no multi-node, no RCCL tuning needed. The
  2B model at bf16 is ~4.6 GB, so the card is nowhere near capacity — which is exactly why QLoRA
  buys nothing here.
- **192 GB VRAM / 240 GB RAM.** The ROCm stack and the harness's dataframes both fit; there is no
  reason to shard or stream.
- **Scratch disk is 5 TB and non-persistent.** Useful for the HF cache during a run, worthless
  after one.

---

## 7. Teardown checklist

Run in order. Steps 1–3 are the ones that are easy to skip and expensive to skip.

1. **Confirm every artifact is off the droplet.** `sync_back.sh --push-hub --to-laptop --tags
   amd1-A,amd1-B,amd1-AB,amd1-base`. Check the private dataset repo on the Hub has the adapters and
   `data/runs/<tag>/`, and check the laptop's `smol-ladder-out/` has the same tree.
2. **Confirm the results are readable**, not just present: each tag's `summary_test.json` exists and
   the pass rates are not all 0.00 (a zero-everywhere sweep is the signature of the tool-call
   failure in §5 step 6).
3. **Verify against the Hub**, which is the copy that survived: pull one adapter back and load it.
4. **Destroy the droplet.** Mode 1: the console. Mode 2: `python ops/amd/driver.py --mode api
   --destroy --droplet smol-ladder`. This is the step that stops billing — powering off does not.
5. **Confirm billing stopped.** The droplet list is empty; the Billing → Credits balance is no
   longer moving.
6. **Delete the snapshot** if you took one for pause/resume — snapshots bill at $0.06/GB/month and
   the AMD credit explicitly does not cover them.
7. **Revoke `AMD_CLOUD_API_TOKEN`** if Mode 2 was used, and remove it from `.env`.
8. **Delete the volumes**, if any were attached: $0.10/GiB/month, billed whether attached or not.
9. **Record what was spent and what it produced**, next to the estimates in §3. That comparison is
   the input to every future AMD estimate; the runbook's numbers are only as good as the first
   measurement that replaces them.

---

## 8. Still unknown — the owner must check these in the console

| # | question | where |
|---|---|---|
| 1 | **Has the $100 credit been applied, and when?** Its deposit date sets the 30-day expiry. | Billing → **Credits** tab (the main balance shows $0.00 and is not the answer) |
| 2 | **Is the price $2.59/h or $1.99/h?** This runbook uses $2.59; the plan uses $1.99. It is a 23% difference in how much work the credit buys. | the droplet creation screen |
| 3 | **Is a payment method on file?** Required to use the credit, and it means an overrun becomes a real charge. | Billing → Payment |
| 4 | **Which ROCm-ready images are actually offered right now**, and what is in the preloaded `rocm` container (ROCm version, torch version)? | the droplet creation screen; then `docker exec -it rocm bash` and `rocminfo` |
| 5 | **Is there an ADC API-tokens page**, and does an ADC PAT behave like a normal DigitalOcean PAT? | ADC console → account/API |
| 6 | **How large is the `test` split's cached-input set?** ~250 tasks × 5 rungs drives the eval estimate; the actual denominator changes the estimate. | `data/inputs`, and `data/skipped_test.json` |
| 7 | **Which vLLM does the preloaded container serve, and does it support Qwen3.5 + LoRA + `qwen3_coder`?** | smoke-test steps 1–7 |
| 8 | **Do arm A's real step time and arm A's real eval time match §3?** They replace the estimates for B and A+B. | after the first two hours |

---

## Sources

- [AMD Developer Cloud (product + FAQ)](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
- [How to Get Started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
- [AMD Developer Cloud GPU Pricing](https://docs.digitalocean.com/products/amd/details/pricing/)
- [How to Create AMD GPU Droplets](https://docs.digitalocean.com/products/amd/how-to/create/)
- [AMD Developer Cloud Limits](https://docs.digitalocean.com/products/amd/details/limits/)
- [AMD Developer Cloud Credits](https://docs.digitalocean.com/products/amd/details/credits/)
- [AMD Developer Cloud GPU Features](https://docs.digitalocean.com/products/amd/details/features/)
- [AMD Developer Cloud GPU Availability](https://docs.digitalocean.com/products/amd/details/availability/)
- [How to Use the Scratch Disk on ADC GPU Droplets](https://docs.digitalocean.com/products/amd/how-to/use-scratch-disk/)
- [Droplet Pricing](https://docs.digitalocean.com/products/droplets/details/pricing/)
- [How to Destroy a Droplet](https://docs.digitalocean.com/products/droplets/how-to/destroy/)
- [Droplet Actions API Reference](https://docs.digitalocean.com/reference/api/reference/droplet-actions/)
- [Bandwidth Billing](https://docs.digitalocean.com/platform/billing/bandwidth/)
- [Volumes Pricing](https://docs.digitalocean.com/products/volumes/details/pricing/)
- [Snapshots Pricing](https://docs.digitalocean.com/products/snapshots/details/pricing/)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [rocm/pytorch tags](https://hub.docker.com/r/rocm/pytorch) · [rocm/vllm tags](https://hub.docker.com/r/rocm/vllm)
- [vLLM v0.30.0 release notes (ROCm wheel and image)](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
- [bitsandbytes: multi-backend (non-CUDA) support](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends)
- [Accelerating LLMs with Flash Attention on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [Claiming the $100 AMD Developer Cloud Credit (third-party)](https://lilting.ch/en/articles/amd-developer-cloud-credit-journey)
