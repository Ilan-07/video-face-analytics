"""Milestone 3: the one place this repo talks to a network LLM.

Every other Milestone 3 module calls `generate()` and nothing else. That keeps the
credential, the retry policy and the response cache in a single auditable file.

Backend is OpenRouter, whose endpoint is OpenAI-compatible, so plain `requests`
suffices -- no SDK. The default model (config.NARRATE_MODEL) is Gemma 4 31B: a
262k context window, Apache-2.0, image-capable, and free.

Reproducibility
---------------
Milestones 1 and 2 have the property that a fresh clone can rebuild and re-verify
every reported number offline. A hosted LLM would break that, so every response is
cached to data/llm_cache/<sha256>.json and the cache is the ONLY part of data/
that is committed to git (see .gitignore). Consequences:

    cache hit   -> returns instantly, no network, no API key required
    cache miss  -> needs OPENROUTER_API_KEY, calls the API, writes the cache

So `pytest` and `eval_story.py` replay Gemma's exact answers with no credential,
while regenerating from scratch is a deliberate act. temperature=0 and a fixed
seed mean a regeneration should reproduce the cache rather than diverge from it.
"""
import base64
import hashlib
import json
import os
import random
import re
import time

import config
import util

log = util.get_logger()


def extract_json_array(text: str):
    """Pull the first top-level JSON array out of an LLM reply, tolerating the
    ```json fences and the chatty preamble small models like to add. Returns None
    when nothing parses. Pure -- unit tested.

    Shared by narrate.py (the event timeline) and describe_scenes.py (batched
    keyframe descriptions), the two places we ask for machine-readable output."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("[")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(candidate[start:], start=start):
        depth += (ch == "[") - (ch == "]")
        if depth == 0:
            try:
                return json.loads(candidate[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None

_ENDPOINT = "/chat/completions"
_KEY_ENV = "OPENROUTER_API_KEY"
_ENV_FILE = config.ROOT / ".env"

_NO_KEY_MSG = (
    f"{_KEY_ENV} is not set, and this prompt is not in the response cache "
    f"({{cache}}).\n"
    "  Milestone 3 generation needs an OpenRouter key (free, no card):\n"
    "    1. create one at https://openrouter.ai/keys\n"
    "    2. enable free endpoints at https://openrouter.ai/settings/privacy\n"
    f"    3. echo '{_KEY_ENV}=sk-or-...' >> .env      (gitignored)\n"
    f"       ...or export {_KEY_ENV} in your shell\n"
    "  Replaying the committed cache needs no key -- this prompt simply isn't in it."
)


def api_key() -> str | None:
    """The OpenRouter key, from the environment or a gitignored .env file.

    The .env fallback exists because the key must never be pasted into a shell
    transcript or committed. Deliberately minimal -- no python-dotenv dependency,
    no interpolation, no export syntax: KEY=value lines and # comments."""
    key = os.environ.get(_KEY_ENV)
    if key:
        return key.strip()
    if not _ENV_FILE.exists():
        return None
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == _KEY_ENV:
            return value.strip().strip("'\"") or None
    return None


def _image_digest(path) -> str:
    """Content hash of an image, so the cache key changes if the frame changes.

    The downscale size is folded in: re-encoding at a different resolution is a
    different request and must not silently reuse the old cached answer."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    return f"{digest}@{config.NARRATE_IMAGE_MAX_SIDE}"


def _encode_image(path) -> str:
    """Base64 JPEG, downscaled so the longest side is NARRATE_IMAGE_MAX_SIDE.

    Gemma 4 resizes to 896px internally, so shipping the 1280px originals just
    inflates the payload -- and a smaller body is markedly likelier to get past a
    rate-limited free endpoint."""
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        limit = config.NARRATE_IMAGE_MAX_SIDE
        if longest > limit:
            scale = limit / longest
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def cache_key(model: str, prompt: str, image_digests: list, params: dict) -> str:
    """Stable cache key for one request. Pure -- unit tested for stability across
    dict ordering, which is why `sort_keys=True` is not optional here."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": list(image_digests),
        "params": params,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str):
    return config.LLM_CACHE_DIR / f"{key}.json"


def _read_cache(key: str):
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)["response"]
    except (json.JSONDecodeError, KeyError, OSError):
        log.warning("corrupt cache entry %s -- regenerating", path.name)
        return None


def _write_cache(key: str, model, prompt, params, images, response) -> None:
    with open(_cache_path(key), "w") as f:
        json.dump({"model": model, "prompt": prompt, "params": params,
                   "images": images, "response": response}, f, indent=2)


def _content(prompt: str, images: list) -> list:
    """OpenAI-style multimodal message content. Images are inlined as data URIs
    because OpenRouter will not reach back into our filesystem."""
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for path in images:
        b64 = _encode_image(path)
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return parts


def _post(model: str, prompt: str, images: list, params: dict) -> str:
    import requests   # deferred: cache replays must not need the dependency

    key = api_key()
    if not key:
        raise RuntimeError(_NO_KEY_MSG.format(cache=config.LLM_CACHE_DIR))

    body = {"model": model,
            "messages": [{"role": "user", "content": _content(prompt, images)}],
            **params}
    delay = 2.0
    for attempt in range(1, config.NARRATE_MAX_RETRIES + 1):
        r = requests.post(
            config.NARRATE_BASE_URL + _ENDPOINT,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json=body, timeout=config.NARRATE_TIMEOUT)
        # ":free" models 404 with "No endpoints found" until the account opts in
        # to prompt publication -- a confusing error worth translating.
        if r.status_code == 404 and ":free" in model:
            raise RuntimeError(
                f"OpenRouter has no endpoint for '{model}'. Free models require "
                "opting in at https://openrouter.ai/settings/privacy "
                "(enable free endpoints that may publish prompts).")
        # 429 = rate limit (the free vision endpoints are congested upstream, and
        # those rejections do NOT consume our daily quota); 5xx = transient.
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == config.NARRATE_MAX_RETRIES:
                raise RuntimeError(
                    f"OpenRouter returned {r.status_code} after "
                    f"{attempt} attempts: {r.text[:200]}")
            wait = delay + random.uniform(0, 1.5)   # jitter: avoid lockstep retries
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = min(int(retry_after), config.NARRATE_RETRY_MAX_DELAY)
            log.warning("HTTP %d from OpenRouter -- retry %d/%d in %.0fs",
                        r.status_code, attempt, config.NARRATE_MAX_RETRIES, wait)
            time.sleep(wait)
            delay = min(delay * 2, config.NARRATE_RETRY_MAX_DELAY)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")

        data = r.json()
        if "choices" not in data:      # OpenRouter reports some errors in a 200
            raise RuntimeError(f"unexpected OpenRouter response: {str(data)[:300]}")
        return data["choices"][0]["message"]["content"]
    raise RuntimeError("unreachable")


def generate(prompt: str, images=None, model: str | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             use_cache: bool | None = None) -> str:
    """Return the model's completion for `prompt` (+ optional image paths).

    Consults data/llm_cache first; only a miss touches the network, and only a
    miss needs OPENROUTER_API_KEY."""
    config.ensure_dirs()
    images = [str(p) for p in (images or [])]
    model = model or config.NARRATE_MODEL
    use_cache = config.NARRATE_CACHE if use_cache is None else use_cache
    params = {
        "max_tokens": max_tokens or config.NARRATE_MAX_TOKENS,
        "temperature": (config.NARRATE_TEMPERATURE
                        if temperature is None else temperature),
        "seed": config.NARRATE_SEED,
    }

    digests = [_image_digest(p) for p in images]
    key = cache_key(model, prompt, digests, params)
    if use_cache:
        hit = _read_cache(key)
        if hit is not None:
            log.info("llm cache hit %s (%d chars)", key[:12], len(hit))
            return hit

    log.info("llm call %s model=%s prompt=%d chars images=%d",
             key[:12], model, len(prompt), len(images))
    response = _post(model, prompt, images, params)
    _write_cache(key, model, prompt, params, digests, response)
    return response


def is_cached(prompt: str, images=None, model: str | None = None,
              max_tokens: int | None = None,
              temperature: float | None = None) -> bool:
    """Whether generate() would answer this from cache. Lets run_pipeline skip the
    narration stages with a warning instead of crashing when there is no key."""
    images = [str(p) for p in (images or [])]
    params = {
        "max_tokens": max_tokens or config.NARRATE_MAX_TOKENS,
        "temperature": (config.NARRATE_TEMPERATURE
                        if temperature is None else temperature),
        "seed": config.NARRATE_SEED,
    }
    key = cache_key(model or config.NARRATE_MODEL, prompt,
                    [_image_digest(p) for p in images], params)
    return _cache_path(key).exists()


def have_key() -> bool:
    return bool(api_key())
