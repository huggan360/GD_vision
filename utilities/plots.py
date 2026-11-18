from pathlib import Path


def plot_training_loss(history, output):
    """Save an epoch-versus-loss plot from the rows in history.json."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    if not history:
        raise ValueError("No training history to plot.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o", color="#2563eb",)
    axis.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    figure.savefig(output, dpi=150)
    return output
