import random
from matplotlib import animation
import numpy as np
import osmnx as ox
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde

dt = 0.000004
normal_offset = 4 * dt

n_agents = 60


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
        self.target_point: np.array = None
        self.offset = 0
        self.status = "walking"


bbox = (47.0478, 47.0459, 8.3047, 8.3080)
area = ox.graph_from_bbox(bbox=bbox, network_type="drive")
buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
roads = ox.features_from_bbox(
    bbox=bbox, tags={"highway": ["residential", "primary"]})
paths = ox.features_from_bbox(
    bbox=bbox, tags={"highway": ["pedestrian", "footway"]})
parks = ox.features_from_bbox(
    bbox=bbox, tags={"leisure": "park", "landuse": "grass"})

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
        v.append(Node(row.Index, np.array([row.x, row.y]), adj[row.Index]))
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
    normal = np.array([-dir[1], dir[0]])
    normal = normal / LA.norm(normal)

    dir = dir / LA.norm(dir) * np.random.uniform(0.7, 1.2)
    my_x += normal * normal_offset

    agent = Agent(i, my_x, dir)
    agent.edge = edge
    agent.offset = normal * normal_offset
    agent.target_node = target
    agent.target_point = (
        target.x + normal * normal_offset - dir / LA.norm(dir) * normal_offset
    )

    agents.append(agent)

fig, ax = plt.subplots()

# set ax color to transparent
ax.patch.set_facecolor("none")

parks.plot(ax=ax, facecolor="green")
buildings.plot(ax=ax, facecolor="silver", alpha=0.7)

# animate the agents

# remove the axes and frame
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

event_node = None

fig2, ax2 = plt.subplots()

# make a grid to compute density to show in a heatmap
x, y = np.zeros(n_agents), np.zeros(n_agents)
counts = []
outfile = open("res.csv", "w")


def update(frame):
    global event_node, adjacent_edges
    # print("Frame", frame)
    ax.clear()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    parks.plot(ax=ax, facecolor="green")
    buildings.plot(ax=ax, facecolor="silver", alpha=0.7)
    roads.plot(ax=ax, color="black", alpha=0.7, linewidth=1)
    paths.plot(ax=ax, color="gray", alpha=1, linewidth=0.7)

    count = 0

    for i, agent in enumerate(agents):
        ax.plot(
            agent.x[0], agent.x[1],
            "bo", zorder=3, markersize=1)
        if agent.status == "waiting":
            continue
        agent.x += agent.v * dt
        # compute the density of agents
        x[i] = agent.x[0]
        y[i] = agent.x[1]
        outfile.write(f"{frame},{agent.id},{agent.x[0]},{agent.x[1]}\n")
        speed = LA.norm(agent.v)
        agent.v = (agent.target_point - agent.x) / \
            LA.norm(agent.target_point - agent.x) * speed

        if LA.norm(agent.x - v[6].x) < 0.0001:
            count += 1
        # # check if the agent is close to a node
        # close_to_node = False
        # for w in v:
        #     if (agent.x - w.x).norm() < 0.0001:
        #         close_to_node = True
        #         if agent.status == "turning":
        #             break
        #         # find the direction to the next node
        #         if len(w.adjacent) == 1:
        #             agent.v = Vector2(0, 0)
        #             agent.status = "waiting"
        #         next_node = random.choice([x for x in v if x.id in w.adjacent])
        #         dir = next_node.x - w.x
        #         dir_len = dir.norm()
        #         agent.x = w.x
        #         agent.v = dir / dir_len * random.uniform(0.7, 1.2)
        #         normal = Vector2(-dir.y, dir.x)
        #         normal = normal / normal.norm()
        #         agent.v += normal * normal_offset
        #         agent.edge = edge_from_node_ids([w.id, next_node.id])
        #         agent.status = "turning"
        # if not close_to_node:
        #     agent.status = "walking"

        # check if the agent is close to the target
        if LA.norm(agent.x - agent.target_point) < normal_offset * 1.5:
            if agent.status == "walking":
                agent.status = "turning"
                next_node = random.choice(
                    [x for x in v if x.id in agent.target_node.adjacent]
                )
                next_dir = next_node.x - agent.target_node.x
                next_normal = np.array([-next_dir[1], next_dir[0]])
                next_normal = next_normal / LA.norm(next_normal)
                corner = (
                    agent.target_node.x
                    + next_normal * normal_offset
                    + next_dir / LA.norm(next_dir) * normal_offset
                )
                agent.target_point = corner
                agent.edge = edge_from_node_ids(
                    [agent.target_node.id, next_node.id])
                agent.target_node = next_node
                agent.v = agent.target_point - agent.x
                agent.v = agent.v / LA.norm(agent.v) * random.uniform(0.7, 1.2)
            elif agent.status == "turning":
                agent.status = "walking"
                agent.target_point = (
                    agent.target_node.x + agent.offset - agent.v / LA.norm(agent.v) * normal_offset)
                agent.v = agent.target_point - agent.x
                agent.v = agent.v / LA.norm(agent.v) * random.uniform(0.7, 1.2)
            else:
                agent.status = "waiting"
                agent.v = np.array([0, 0])

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
    if frame >= 10:
        # ax.plot(
        #     event_node.x.x,
        #     event_node.x.y,
        #     marker="x",
        #     color="r",
        #     markersize=10,
        #     linewidth=15,
        # )
        pass

        # plot the graph
    for w in v:
        ax.plot(w.x[0], w.x[1], "ko", alpha=0.2)
        for adj in w.adjacent:
            w2 = next((x for x in v if x.id == adj), None)
            ax.plot([w.x[0], w2.x[0]], [w.x[1], w2.x[1]], "k-", alpha=0.2)

    # plot the heatmap
    k = gaussian_kde(np.vstack([x, y]), bw_method=0.1)
    xi, yi = np.mgrid[bbox[2]:bbox[3]:100j, bbox[0]:bbox[1]:100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # normalize the heatmap
    zi = (zi - zi.min()) / (zi.max() - zi.min())

    # add levels so that 0 density is transparent
    ax.contourf(
        xi,
        yi,
        zi.reshape(xi.shape),
        levels=np.linspace(
            0.1, 1, 10),
        alpha=0.7
    )

    # new subplot
    counts.append(count)
    ax2.plot(pd.Series(counts).rolling(10).mean(), color="orange")
    ax2.set_xticklabels([])

    return ax


ani = FuncAnimation(fig, update, frames=200, blit=False, repeat=True)


# save as apng
ani.save("output.gif", fps=20, bitrate=-1,
         # make figure background transparent
         savefig_kwargs={"transparent": True, "facecolor": "none"},
         )


outfile.close()
# ox.plot_graph(area, ax=ax, node_color="r", node_zorder=3, show=False)

# plt.show()
