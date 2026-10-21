"""Generate the notebooks in notebooks/ (serve + sharp-spark-128k, Colab + Kaggle).

Keeping the notebooks generated from one script means the two platforms never
drift apart, and the committed .ipynb files always have empty outputs — no
tunnel URL and no API key ever land in git.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_URL = "https://github.com/iramarfalcao/gpu-free-serve.git"
OUT = Path(__file__).resolve().parent.parent / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def secrets_cell(platform: str) -> dict:
    if platform == "colab":
        reader = """
from google.colab import userdata

def read_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None
"""
        where = "Colab: click the key icon in the left sidebar and add them there."
    else:
        reader = """
from kaggle_secrets import UserSecretsClient

_client = UserSecretsClient()

def read_secret(name):
    try:
        return _client.get_secret(name)
    except Exception:
        return None
"""
        where = "Kaggle: Add-ons > Secrets (and turn Internet ON in the right sidebar)."

    return code(
        f"""
# Secrets are read from the platform's secret manager and never written into
# this notebook. {where}
#
#   GITHUB_TOKEN     scope `gist` -> lets `gpufree connect --gist` work
#   HF_TOKEN         only for gated models (Llama, Gemma, ...)
#   NGROK_AUTHTOKEN  only if TUNNEL = "ngrok"
#   GPUFREE_API_KEY  optional fixed API key; otherwise a random one is generated
import os
{reader.strip()}

for name in ("GITHUB_TOKEN", "HF_TOKEN", "NGROK_AUTHTOKEN", "GPUFREE_API_KEY"):
    value = read_secret(name)
    if value:
        os.environ[name] = value
        print(f"{{name}}: loaded")
    else:
        print(f"{{name}}: not set (fine if you don't need it)")
"""
    )


def build(platform: str) -> dict:
    workdir = "/content" if platform == "colab" else "/kaggle/working"
    accel = (
        "Runtime > Change runtime type > T4 GPU"
        if platform == "colab"
        else "Settings > Accelerator > GPU T4 x2 (and Internet: On)"
    )
    title = "Colab" if platform == "colab" else "Kaggle"

    cells = [
        md(
            f"""
# gpu-free-serve — {title}

Boots an LLM on this notebook's free GPU and exposes an OpenAI-compatible
endpoint you can call from your own machine.

**Before running:** {accel}

Then run the cells top to bottom. The last cell keeps running — that is the
server. Closing this tab kills the endpoint.
"""
        ),
        code("!nvidia-smi"),
        code(
            f"""
# Clone the repo (or update it if it is already here).
import os, subprocess

os.chdir("{workdir}")
if os.path.isdir("gpu-free-serve"):
    subprocess.run(["git", "-C", "gpu-free-serve", "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
os.chdir("{workdir}/gpu-free-serve")
print(os.getcwd())
"""
        ),
        secrets_cell(platform),
        md(
            """
## Pick a model

The catalog below comes from `models.yaml` in the repo. Edit that file (and push)
to add your own models — or set `MODEL` to a raw id:

* `Qwen/Qwen2.5-7B-Instruct-AWQ` → served by vLLM
* `ollama:llama3.1:8b` → served by Ollama
"""
        ),
        code("!python -m gpufree.server.serve --list"),
        code(
            """
# Some images (Kaggle today) ship a PyTorch built for a newer CUDA toolkit than
# their torchaudio, so `import torchaudio` raises "compiled with different CUDA
# versions" and takes vLLM down with it: vllm -> transformers ->
# transformers.audio_utils -> torchaudio. Serving a text model needs no audio,
# and transformers only imports torchaudio when it is present, so drop it — but
# only when it is actually broken, so healthy images keep working.
import importlib.util, subprocess, sys

def _torchaudio_broken():
    if importlib.util.find_spec("torchaudio") is None:
        return False
    try:
        subprocess.run(
            [sys.executable, "-c", "import torchaudio"],
            check=True, capture_output=True, timeout=180,
        )
        return False
    except subprocess.CalledProcessError:
        return True

if _torchaudio_broken():
    print("torchaudio does not match this PyTorch build — removing it (vLLM needs no audio).")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

print("torchaudio installed:", importlib.util.find_spec("torchaudio") is not None)
"""
        ),
        code(
            f"""
MODEL = "qwen2.5-7b-awq"   # a catalog name, `org/model`, or `ollama:tag`
TUNNEL = "cloudflare"      # "cloudflare" (no account) or "ngrok" (needs authtoken)
PUBLISH = "gist"           # "gist" publishes the endpoint; "none" just prints it

# This cell blocks while the server is alive. First run installs vLLM and
# downloads the weights (10-15 min).
!python -m gpufree.server.serve --model {{MODEL}} --tunnel {{TUNNEL}} --publish {{PUBLISH}}
"""
        ),
        md(
            """
## Use it from your machine

```bash
pipx install git+https://github.com/iramarfalcao/gpu-free-serve.git   # once

gpufree connect --gist          # or: gpufree connect --url <url> --key <key>
gpufree status
gpufree chat "explain attention in two sentences"
gpufree proxy                   # http://127.0.0.1:8787/v1 — no key needed
```

Point any OpenAI client at the proxy:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-needed")
```

**Do not commit this notebook with its output** — the banner prints the API key.
Edit > Clear all outputs first.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# sharp-spark-128k: a GGUF-only community model imported into Ollama at 128k.
#
# vLLM cannot serve it (the repo ships no config.json, and the base model is a
# custom architecture), so this notebook creates a local Ollama model from the
# GGUF instead of pulling one. The model fits one T4 with a 128k context.
#
# Pinned to the fork while the local-import and streaming fixes are unmerged;
# override GPUFREE_SHARP_SPARK_REPO to build it against upstream instead.
SHARP_SPARK_REPO = os.environ.get(
    "GPUFREE_SHARP_SPARK_REPO", "https://github.com/mufflegyro/gpu-free-serve.git"
)

SHARP_GGUF_REPO = "peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF"
SHARP_GGUF_FILE = "Sharp-Spark-X2.5-4B-Q5_K_XL.gguf"
SHARP_MODEL = "sharp-spark-q5-128k"
SHARP_CTX = 131072


def build_sharp_spark(platform: str) -> dict:
    workdir = "/content" if platform == "colab" else "/kaggle/working"
    accel = (
        "Runtime > Change runtime type > T4 GPU"
        if platform == "colab"
        else "Settings > Accelerator > GPU T4 x2 (and Internet: On)"
    )
    title = "Colab" if platform == "colab" else "Kaggle"

    def fill(text: str) -> str:
        return (
            text.strip()
            .replace("__WORKDIR__", workdir)
            .replace("__REPO__", SHARP_SPARK_REPO)
            .replace("__GGUF_REPO__", SHARP_GGUF_REPO)
            .replace("__GGUF_FILE__", SHARP_GGUF_FILE)
            .replace("__MODEL__", SHARP_MODEL)
            .replace("__CTX__", str(SHARP_CTX))
            .replace("__ACCEL__", accel)
            .replace("__TITLE__", title)
        )

    cells = [
        md(
            fill(
                """
# gpu-free-serve — Sharp-Spark-X2.5-4B at 128k (__TITLE__)

Boots [peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF](https://huggingface.co/peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF)
on this notebook's free GPU and exposes an OpenAI-compatible endpoint.

**Before running:** __ACCEL__

Why this notebook exists: that repo contains **only `.gguf` files** — no
`config.json`, no tokenizer — and its base model `XHToken/Spark-X2.5-4B` is a
custom architecture (`Spark2_5ForCausalLM`). vLLM rejects both, so this is the
Ollama (llama.cpp) route: download the GGUF, `ollama create` it with a 128k
context, then serve it.

The context is cheap here: the model is native to 1M positions, only 9 of its
36 layers are full attention (27 are sliding-window at 512 tokens) and it has 4
KV heads, so a 128k cache costs ~4.6 GiB f16 — about 7.6 GiB total with the
weights, i.e. one T4.
"""
            )
        ),
        code("!nvidia-smi"),
        code(
            fill(
                """
# Clone the repo (or update it if it is already here).
import os, subprocess

os.chdir("__WORKDIR__")
if os.path.isdir("gpu-free-serve"):
    subprocess.run(["git", "-C", "gpu-free-serve", "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "__REPO__"], check=True)
os.chdir("__WORKDIR__/gpu-free-serve")
print(os.getcwd())
"""
            )
        ),
        secrets_cell(platform),
        md(
            """
## Import the GGUF into Ollama

Downloads ~3 GiB of weights, then creates a local model with `num_ctx` baked
into it (so the context survives restarts and applies to every request).

`ollama create` needs a running daemon, so this starts a throwaway one on port
11435 — the daemon gpufree starts in the serve cell owns 11434, and two on one
port is an "address already in use" failure.
"""
        ),
        code(
            fill(
                """
import os, shutil, subprocess, time, urllib.request

work = "__WORKDIR__"
model_name = "__MODEL__"
num_ctx = __CTX__

# 1. Ollama itself (the serve cell would install it later; we need it now).
if not shutil.which("ollama"):
    if not shutil.which("zstd"):
        subprocess.run("apt-get -qq update && apt-get -qq install -y zstd", shell=True, check=False)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

# 2. Weights.
from huggingface_hub import hf_hub_download
gguf = hf_hub_download("__GGUF_REPO__", "__GGUF_FILE__", local_dir=os.path.join(work, "gguf"))
print("gguf:", gguf)

# 3. Modelfile: the context lives here, not in a cell we have to remember to re-run.
modelfile = os.path.join(work, "Modelfile.sharp-spark")
with open(modelfile, "w") as fh:
    print("FROM", gguf, file=fh)
    print("PARAMETER num_ctx", num_ctx, file=fh)
    print("PARAMETER num_batch 1024", file=fh)

# 4. Throwaway daemon on our own port, then create + inspect + stop.
env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11435")
logpath = os.path.join(work, "ollama-import.log")
log = open(logpath, "w")
daemon = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(90):
        try:
            urllib.request.urlopen("http://127.0.0.1:11435/api/tags", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("ollama did not start on 11435 — see " + logpath)
    subprocess.run(["ollama", "create", model_name, "-f", modelfile], env=env, check=True)
    subprocess.run(["ollama", "show", model_name], env=env, check=False)
finally:
    daemon.terminate()
    try:
        daemon.wait(timeout=30)
    except subprocess.TimeoutExpired:
        daemon.kill()
    log.close()

# `ollama show` above prints a TEMPLATE section when the GGUF carries a chat
# template. If it comes back empty, append the repo's chat_template.jinja to the
# Modelfile and re-run `ollama create` — otherwise chat formatting is generic
# and this model's prompt template is wasted.
"""
            )
        ),
        md(
            """
## Serve it

The env below is what makes 128k usable:

* `OLLAMA_KEEP_ALIVE=-1` — never unload, so a long prompt is prefilled once and
  its KV/prefix is reused across requests instead of re-allocated each time.
* `OLLAMA_NUM_PARALLEL=1` — anything higher splits the 128k window across slots.
* `OLLAMA_KV_CACHE_TYPE=q8_0` — halves the 128k cache (4.6 → 2.3 GiB) and the
  bandwidth attention reads per token. Drop it for f16 if you want the extra
  precision; it still fits.

Verify the context actually took from the log printed by the next cell:
`llama_context: n_ctx = 131072` and two `llama_kv_cache_iswa` lines (one sized
to 131072, one to 512). If `n_ctx` is smaller, the fit was reduced — lower
`num_ctx` in the Modelfile.
"""
        ),
        code(
            fill(
                """
import os

os.environ.update(
    OLLAMA_KEEP_ALIVE="-1",
    OLLAMA_FLASH_ATTENTION="1",
    OLLAMA_KV_CACHE_TYPE="q8_0",
    OLLAMA_NUM_PARALLEL="1",
    OLLAMA_MAX_LOADED_MODELS="1",
)

MODEL = "ollama:__MODEL__"
TUNNEL = "cloudflare"      # "cloudflare" (no account) or "ngrok" (needs authtoken)
PUBLISH = "gist"           # "gist" publishes the endpoint; "none" just prints it

# This cell blocks while the server is alive. First run also loads the model.
!python -m gpufree.server.serve --model {MODEL} --tunnel {TUNNEL} --publish {PUBLISH}
"""
            )
        ),
        md(
            """
## Use it from your machine

```bash
pipx install --force git+<the repo you cloned above>   # once
gpufree connect --gist                                 # or: --url <url> --key <key>
gpufree chat "summarize this file" < notes.md
gpufree proxy                                          # http://127.0.0.1:8787/v1
```

Long prompts are the point of this notebook: prefill of a genuinely 128k-token
prompt costs a minute or two on a T4, then keeps generating at full speed — the
sliding-window layers mean per-token attention stays cheap even at that length.

**Two warnings.** Kaggle meters GPU-hours whether or not the model is busy, so
stop the session when you are done (closing the tab does not stop it). And this
notebook's output contains your API key — Edit > Clear all outputs before any
Save Version.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for platform in ("colab", "kaggle"):
        path = OUT / f"{platform}_serve.ipynb"
        path.write_text(json.dumps(build(platform), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path}")
        path = OUT / f"{platform}_sharp_spark_128k.ipynb"
        path.write_text(json.dumps(build_sharp_spark(platform), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
