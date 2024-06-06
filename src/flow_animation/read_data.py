import random
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde

dt = 0.00001
normal_offset = 4 * dt

n_agents = 200

class Node:
    def __init__(self, id, x, adjacent):
        self.id = id
        self.x = x
        self.adjacent = adjacent


class Agent:
    def __init__(self, id, x, v):
        self.id = id
        self.x = x
        self.v = v
        self.edge = None
        self.target_node: Node = None
        self.target_point: Vector2 = None
        self.offset = 0
        self.status = "walking"


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vector2(self.x / other, self.y / other)

    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5

    def dot(self, other):
        return self.x * other.x + self.y * other.y

bbox = (47.0478, 47.0459, 8.3047, 8.3080)
area = ox.graph_from_bbox(bbox=bbox, network_type="drive")
buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
parks = ox.features_from_bbox(bbox=bbox, tags={"leisure": "park", "landuse": "grass"})

adj = dict()
for i in area.adjacency():
    adj[i[0]] = set(i[1].keys())

# make adj symmetric
for i in adj:
    for j in adj[i]:
        adj[j].add(i)

node_df, edge_df = ox.graph_to_gdfs(area)

v = []
for row in node_df.itertuples():
    if row.Index in adj and len(adj[row.Index]) > 0:
        v.append(Node(row.Index, Vector2(row.x, row.y), adj[row.Index]))
e = []
for row in edge_df.itertuples():
    e.append({row.Index[0], row.Index[1]})


def node_from_id(idx):
    for i in v:
        if i.id == idx:
            return i
    return None


def edge_from_node_ids(idx):
    for i in e:
        if idx[0] in i and idx[1] in i:
            return i
    return None


def point_on_edge(edge):
    n1 = node_from_id(list(edge)[0])
    n2 = node_from_id(list(edge)[1])
    return (n1.x - n2.x) * np.random.rand() + n2.x

def edge_geometry(edge):
    n1 = node_from_id(list(edge)[0])
    n2 = node_from_id(list(edge)[1])
    return (n1.x.x, n2.x.x), (n1.x.y, n2.x.y)

agents = []
for i in range(n_agents):
    edge = random.choice(e)
    p = point_on_edge(edge)
    l = list(edge)
    v1 = node_from_id(l[0]).x
    v2 = node_from_id(l[1]).x

    my_x = p
    target = node_from_id(random.choice(l))
    dir = target.x - my_x
    normal = Vector2(-dir.y, dir.x)
    normal = normal / normal.norm()

    dir = dir / dir.norm() * np.random.uniform(0.7, 1.2)
    my_x += normal * normal_offset

    agent = Agent(i, my_x, dir)
    agent.edge = edge
    agent.offset = normal * normal_offset
    agent.target_node = target
    agent.target_point = (
        target.x + normal * normal_offset - dir / dir.norm() * normal_offset
    )

    agents.append(agent)

#big with small margins
fig, ax = plt.subplots(figsize=(5, 5))

parks.plot(ax=ax, facecolor="green")
buildings.plot(ax=ax, facecolor="silver", alpha=0.7)

# animate the agents

event_node = None

fig2, ax2 = plt.subplots()

# make a grid to compute density to show in a heatmap
x, y = np.zeros(n_agents), np.zeros(n_agents)
counts = []
# df from csv
df = pd.read_csv("res.csv")

def update(frame):
    if frame == 0:
        plt.waitforbuttonpress()
    global event_node, adjacent_edges
    # print("Frame", frame)
    ax.clear()
    parks.plot(ax=ax, facecolor="green")
    buildings.plot(ax=ax, facecolor="silver", alpha=0.7)

    count = 0
    now_df = df[df["frame"] == frame]
    for i, agent in enumerate(agents):
        agent_df = now_df[now_df["id"] == agent.id]
        agent.x = Vector2(agent_df["x"].values[0], agent_df["y"].values[0])
        agent.v = Vector2(agent_df["vx"].values[0], agent_df["vy"].values[0])
        agent.status = agent_df["status"].values[0]
        ax.plot(agent.x.x, agent.x.y, "bo", zorder=3, markersize=1)

        # # compute the density of agents
        x[i] = agent.x.x
        y[i] = agent.x.y

        if (agent.x - v[6].x).norm() < 0.0001:
            count += 1


    if frame >= 10 and frame < 20:
        if frame == 10:
            event_node = random.choice(v)
            adjacent_edges = [
                edge_from_node_ids([event_node.id, x]) for x in event_node.adjacent
            ]
            
        for agent in agents:
            if agent.edge in adjacent_edges:
                agent.target_point = event_node.x
            if frame == 10:
                agent.v *= 1.25
    
    # make a big red ring around node 6
    ax.plot(v[6].x.x, v[6].x.y, "ro", markersize=10, alpha=0.5)

    # plot the graph
    for w in v:
        ax.plot(w.x.x, w.x.y, "ko", alpha=0.5)
        for adj in w.adjacent:
            w2 = next((x for x in v if x.id == adj), None)
            ax.plot([w.x.x, w2.x.x], [w.x.y, w2.x.y], "k-", alpha=0.3)

    # plot the heatmap
    k = gaussian_kde(np.vstack([x, y]), bw_method=0.1)
    xi, yi = np.mgrid[bbox[2]:bbox[3]:100j, bbox[0]:bbox[1]:100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
    
    # new subplot
    counts.append(count)
    ax2.plot(pd.Series(counts).rolling(10).mean(), color="orange")
    ax2.set_xticklabels([])

    return ax


# wait for key press
ready = False
while not ready:
    ready = plt.waitforbuttonpress()

ani = FuncAnimation(fig, update, frames=1000, blit=False)

ox.plot_graph(area, ax=ax, node_color="r", node_zorder=3)

plt.show()
