import gradio as gr
import random
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from evaluate import run_evaluation
from visualize import generate_graph
from src.environment import TrafficEnv
from src.agent import DeterministicAgent

# ---------------------------------------------------------------------------
# Global env state for the OpenEnv REST API
# ---------------------------------------------------------------------------
_env: TrafficEnv = None
_agent: DeterministicAgent = DeterministicAgent()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

api = FastAPI(title="SmartTraffic OpenEnv API", lifespan=lifespan)

# ---------- Pydantic schemas ----------
class ResetRequest(BaseModel):
    seed: int = None

class StepRequest(BaseModel):
    action: int

# ---------- OpenEnv REST endpoints ----------
@api.post("/reset")
async def reset(body: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    global _env, _agent
    seed = None
    if body is not None and body.seed is not None:
        seed = body.seed
    else:
        seed = random.randint(1000, 99999)

    _env = TrafficEnv({
        "max_time": 200,
        "arrival_rate": 2.0,
        "congestion_multiplier": 1.5,
        "emergency_prob": 0.05,
    })
    _agent = DeterministicAgent()
    state = _env.reset(seed=seed)
    return JSONResponse(content={"observation": state.to_dict(), "seed": seed})


@api.post("/step")
async def step(body: StepRequest):
    """Take one environment step and return the result."""
    global _env
    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialised. Call POST /reset first."},
        )
    result = _env.step(body.action)
    return JSONResponse(content={
        "observation": result.state.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    })


@api.get("/observe")
async def observe():
    """Return the current observation without stepping."""
    global _env
    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialised. Call POST /reset first."},
        )
    return JSONResponse(content={"observation": _env.state().to_dict()})


@api.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Gradio UI (mounted on /ui)
# ---------------------------------------------------------------------------
def run_simulation(manual_seed=None):
    try:
        seed = int(manual_seed) if manual_seed and str(manual_seed).isdigit() else random.randint(1000, 99999)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            scores = run_evaluation(base_seed=seed, silent=False)
        logs = f.getvalue()

        graph_path = "optimization_results.png"
        generate_graph(scores, seed, output_path=graph_path)

        return logs, graph_path
    except Exception as e:
        return f"Error running simulation: {str(e)}", None


with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚦 Smart Traffic Optimization Environment (OpenEnv)")
    gr.Markdown("Welcome to the Traffic Simulator. Watch how our AI improves traffic flow in real time.")

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2, min_width=320):
            seed_input = gr.Textbox(label="Optional Seed (Empty for Random)", placeholder="e.g. 42")
            run_btn = gr.Button("🚀 Run Traffic Evaluator", variant="primary", size="lg")
            gr.HTML("""
                <p style='
                    text-align: center;
                    color: rgba(100, 100, 100, 0.75);
                    font-size: 0.88em;
                    margin: 4px 0 16px 0;
                    font-weight: 400;
                    letter-spacing: 0.01em;
                '>Click to simulate AI-driven traffic optimization across difficulty levels.</p>
            """)
        with gr.Column(scale=1):
            pass

    gr.Markdown("""
    <div style='background-color: #ffeaea; border-left: 4px solid #ff4d4f; padding: 12px; margin: 15px 0px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
        <span style='color: #a8071a; font-weight: 600; font-size: 1.05em;'>🚨 Active Protocol:</span>
        <span style='color: #434343;'>System prioritizes emergency vehicles in real-time.</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center; color: #555; margin-bottom: 2px;'>Simulation Logs</h3>")
            output_text = gr.Textbox(show_label=False, lines=22, interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("<h3 style='text-align: center; color: #222; margin-bottom: 10px; border-bottom: 1px solid #eaeaea; padding-bottom: 5px;'>Performance Improvement After Optimization</h3>")
            output_img = gr.Image(show_label=False, type="filepath")

    run_btn.click(fn=run_simulation, inputs=[seed_input], outputs=[output_text, output_img])


# Mount Gradio on /ui so the root path stays clean for the REST API
app = gr.mount_gradio_app(api, interface, path="/")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
