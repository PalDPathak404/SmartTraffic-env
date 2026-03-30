import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from src.environment import TrafficEnv
from src.agent import DeterministicAgent

app = FastAPI(title="SmartTraffic OpenEnv API")

# Global environment state
_env = None
_agent = DeterministicAgent()

# ---- Pydantic Schemas ----
class ResetRequest(BaseModel):
    seed: int = None

class StepRequest(BaseModel):
    action: int

# ---- Standard Endpoints ----
@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "message": "SmartTraffic OpenEnv API is running"}

@app.post("/reset")
@app.post("/openenv/reset")
async def reset_env(body: ResetRequest = None):
    global _env, _agent
    seed = body.seed if body and body.seed is not None else None
    
    # Initialize environment with OpenEnv configurations
    _env = TrafficEnv({
        "max_time": 50,
        "arrival_rate": 2.0,
        "congestion_multiplier": 1.5,
        "emergency_prob": 0.05
    })
    _agent = DeterministicAgent()
    state = _env.reset(seed=seed)
    
    # Extract native dictionary out of the dataclass
    observation = state.to_dict() if hasattr(state, "to_dict") else state
    
    return {
        "observation": observation,
        "info": {"message": "Environment reset successful"}
    }

@app.post("/step")
@app.post("/openenv/step")
async def step_env(body: StepRequest):
    global _env
    if _env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    
    try:
        action = int(body.action)
    except (ValueError, TypeError):
        action = 0  # Fallback to safe action
        
    result = _env.step(action)
    
    observation = result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
    
    return {
        "observation": observation,
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": {"message": "Step executed successfully"}
    }

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=7860)
