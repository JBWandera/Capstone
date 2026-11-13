"""Small plotting helpers for the notebooks."""

from __future__ import annotations

import io


def show_jpeg(fig, quality: int = 80, dpi: int = 90) -> None:
    """Show a figure of photos as JPEG. PNG keeps notebooks with photo grids at
    several MB per figure; JPEG keeps them small enough for git and GitHub."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": quality})
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="jpeg"))
