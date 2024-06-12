import random
from matplotlib import animation
import numpy as np
import osmnx as ox
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde

N_AGENTS = 100

DT = 0.000004


bbox = (47.383445, 47.374291, 8.539601, 8.554085)
area = ox.graph_from_bbox(bbox=bbox, network_type="drive")

# definine which buildings/roads to plot
buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
roads = ox.features_from_bbox(
    bbox=bbox, tags={"highway": ["trunk", "primary", "secondary", "tertiary", "residential", "unclassified", "living_street"]})

node_df, edge_df = ox.graph_to_gdfs(area)

# print(edge_df)

fig, ax = plt.subplots(figsize=(20, 20))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# set ax color to transparent
ax.patch.set_facecolor("none")
buildings.plot(ax=ax, facecolor="silver", alpha=0.7)


def update(frame):
    """
    Main simulation loop
    """

    ax.clear()
    buildings.plot(ax=ax, facecolor="silver", alpha=0.7)

    max_lanes = edge_df["lanes"].max()

    # plot all roads from edge_df
    for row in edge_df.itertuples():
        x, y = row.geometry.xy

        ax.plot(x, y, color="black", alpha=0.7, linewidth=1)


ani = FuncAnimation(fig, update, frames=10, blit=False, repeat=True)

ani.save("output.gif", fps=20, bitrate=-1,
         # make figure background transparent
         savefig_kwargs={"transparent": True, "facecolor": "none"},
         )
