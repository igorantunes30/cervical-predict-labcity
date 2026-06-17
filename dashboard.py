"""
Dashboard web para monitoramento do treino GLSim em tempo real.
Acesse: http://localhost:7860

Lê os logs do TensorBoard gerados durante o treino e exibe:
  - Curvas de loss e accuracy (train/val)
  - Status da época atual
  - Melhor accuracy e ETA
"""

import os
import glob
import time
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


RESULTS_DIR = Path(__file__).parent / "results_train"
REFRESH_INTERVAL = 5  # segundos


def find_latest_run():
    runs = sorted(RESULTS_DIR.glob("*/tensorboard"), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None


def load_tb_scalars(tb_dir):
    if not HAS_TB or not tb_dir or not tb_dir.exists():
        return {}

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    return data


def make_figure(scalars):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss (treino)", "Loss (validação)", "Acc@1 (treino)", "Acc@1 (validação)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    color_train = "#3B82F6"
    color_val = "#10B981"

    def add_trace(tag, row, col, color, name):
        if tag in scalars:
            d = scalars[tag]
            fig.add_trace(
                go.Scatter(x=d["steps"], y=d["values"], mode="lines+markers",
                           name=name, line=dict(color=color, width=2),
                           marker=dict(size=4)),
                row=row, col=col,
            )

    add_trace("train/loss", 1, 1, color_train, "Train Loss")
    add_trace("val/loss",   1, 2, color_val,   "Val Loss")
    add_trace("train/acc1", 2, 1, color_train, "Train Acc@1")
    add_trace("val/acc1",   2, 2, color_val,   "Val Acc@1")

    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=30, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="#334155", title_text="Época")
    fig.update_yaxes(gridcolor="#334155")

    return fig


def get_status(scalars):
    lines = []
    run_dir = find_latest_run()
    run_name = run_dir.parent.name if run_dir else "–"
    lines.append(f"**Run:** `{run_name}`")

    if not scalars:
        lines.append("⏳ Aguardando início do treino...")
        return "\n\n".join(lines)

    if "val/acc1" in scalars:
        vals = scalars["val/acc1"]["values"]
        steps = scalars["val/acc1"]["steps"]
        if vals:
            best_acc = max(vals)
            best_ep = steps[vals.index(best_acc)]
            last_acc = vals[-1]
            last_ep = steps[-1]
            lines.append(f"**Época atual:** {last_ep}")
            lines.append(f"**Val Acc@1 atual:** {last_acc:.2f}%")
            lines.append(f"**Melhor Val Acc@1:** {best_acc:.2f}% (época {best_ep})")

    if "train/loss" in scalars:
        vals = scalars["train/loss"]["values"]
        if vals:
            lines.append(f"**Train Loss:** {vals[-1]:.4f}")

    if "train/lr" in scalars:
        vals = scalars["train/lr"]["values"]
        if vals:
            lines.append(f"**LR atual:** {vals[-1]:.6f}")

    return "\n\n".join(lines)


def update():
    tb_dir = find_latest_run()
    scalars = load_tb_scalars(tb_dir)
    fig = make_figure(scalars)
    status = get_status(scalars)
    return fig, status


def build_app():
    with gr.Blocks(
        title="GLSim - Cervical Cancer Training",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { background: #0f172a; }
        .svelte-1gfkn6j { background: #1e293b; }
        h1 { color: #e2e8f0 !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# GLSim · Cervical Cancer Classification\n"
            "**Dataset:** Herlev (5 classes · 4.049 células) &nbsp;|&nbsp; "
            "**Modelo:** ViT-B/16 + GLSim &nbsp;|&nbsp; "
            f"**Atualiza a cada {REFRESH_INTERVAL}s**"
        )

        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(label="Curvas de Treino", show_label=False)
            with gr.Column(scale=1):
                status_md = gr.Markdown("⏳ Aguardando dados...")

        timer = gr.Timer(value=REFRESH_INTERVAL)
        timer.tick(fn=update, outputs=[plot, status_md])

        demo.load(fn=update, outputs=[plot, status_md])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        inbrowser=True,
    )
