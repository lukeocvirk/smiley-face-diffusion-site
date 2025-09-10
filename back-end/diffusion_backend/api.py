from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import NoisePredictor
from .noise_schedule import LinearNoiseSchedule


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InitRequest(BaseModel):
    n: int = 3000
    ckpt: Optional[str] = None
    seed: Optional[int] = None
    noise_seed: Optional[int] = None


class InitResponse(BaseModel):
    session_id: str
    T: int
    n: int


class StepResponse(BaseModel):
    t: int
    n: int
    points: list[list[float]]
    bounds: dict


SESSIONS: Dict[str, dict] = {}


def _load_model_and_schedule(ckpt_path: Optional[Path], device: torch.device, sched_seed: Optional[int] = None):
    if ckpt_path is None:
        ckpt_path = Path("checkpoints/latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    sched_meta = payload.get("schedule", {}) if isinstance(payload, dict) else {}
    timesteps = int(sched_meta.get("timesteps", 300))
    beta_start = float(sched_meta.get("beta_start", 1e-4))
    beta_end = float(sched_meta.get("beta_end", 2e-2))

    model = NoisePredictor(num_timesteps=timesteps).to(device)
    state_key = "model_ema" if isinstance(payload, dict) and "model_ema" in payload else "model"
    model.load_state_dict(payload[state_key])  # type: ignore[index]
    model.eval()

    schedule = LinearNoiseSchedule(
        timesteps=timesteps, beta_start=beta_start, beta_end=beta_end, device=device, seed=sched_seed
    )
    return model, schedule


def _precompute_traj(n: int, model: NoisePredictor, schedule: LinearNoiseSchedule, device: torch.device,
                     seed: Optional[int]) -> torch.Tensor:
    T = schedule.T
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    x_t = torch.randn((n, 2), device=device, dtype=torch.float32, generator=g)
    traj = torch.empty((T, n, 2), dtype=torch.float32)
    with torch.no_grad():
        for t_idx in range(T - 1, -1, -1):
            traj[t_idx] = x_t.detach().cpu()
            t_batch = torch.full((n,), t_idx, dtype=torch.long, device=device)
            eps_pred = model(x_t, t_batch)
            x_t = schedule.backward(x_t, eps_pred, t_batch)
    return traj


def _compute_bounds(traj: torch.Tensor):
    # traj: [T, N, 2] on CPU
    mn = traj.amin(dim=1)  # [T, 2]
    mx = traj.amax(dim=1)  # [T, 2]
    center = 0.5 * (mn + mx)  # [T, 2]
    extent = (mx - mn)  # [T, 2]
    side = torch.maximum(extent[:, 0], extent[:, 1]).clamp_min(1e-3)  # [T]
    return {
        "cx": center[:, 0].numpy().tolist(),
        "cy": center[:, 1].numpy().tolist(),
        "half": (0.5 * side).numpy().tolist(),
    }


app = FastAPI(title="Diffusion Backend API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/init", response_model=InitResponse)
def api_init(req: InitRequest):
    device = _detect_device()
    ckpt_path = Path(req.ckpt) if req.ckpt else None
    try:
        model, schedule = _load_model_and_schedule(ckpt_path, device, sched_seed=req.noise_seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    traj = _precompute_traj(req.n, model, schedule, device, seed=req.seed)
    bounds = _compute_bounds(traj)

    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "traj": traj,  # CPU tensor [T, n, 2]
        "bounds": bounds,
        "T": int(traj.shape[0]),
        "n": int(traj.shape[1]),
    }
    return InitResponse(session_id=sid, T=SESSIONS[sid]["T"], n=SESSIONS[sid]["n"])


@app.get("/api/meta")
def api_meta(session_id: str = Query(...)):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"T": s["T"], "n": s["n"]}


@app.get("/api/step", response_model=StepResponse)
def api_step(session_id: str = Query(...), t: int = Query(...)):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    T = s["T"]
    if not (0 <= t < T):
        raise HTTPException(status_code=400, detail=f"t must be in [0, {T-1}]")
    traj: torch.Tensor = s["traj"]
    pts = traj[t].numpy().tolist()
    b = {k: s["bounds"][k][t] for k in ("cx", "cy", "half")}
    return StepResponse(t=int(t), n=int(s["n"]), points=pts, bounds=b)


# Entry to run with: uvicorn diffusion_backend.api:app --reload --port 8000
