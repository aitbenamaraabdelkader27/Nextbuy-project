"""Consistent Plotly styling, without external fonts or assets."""

import plotly.graph_objects as go

COLORS = ["#087F8C", "#5263C5", "#F2A65A", "#D2647E"]


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        colorway=COLORS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#172B4D", "size": 12},
        margin={"l": 12, "r": 20, "t": 40, "b": 20},
        legend={"orientation": "h", "y": -0.2},
        title_font_size=17,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF5", zeroline=False)
    return fig
