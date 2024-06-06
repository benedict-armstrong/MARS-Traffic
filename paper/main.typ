#import "@preview/charged-ieee:0.1.0": ieee

#show: ieee.with(
  title: [Game-theoretic Multi-Agent Reinforcement Learning Simulation of Traffic],
  abstract: [
    In this project we aim to explore the application of multi-agent reinforcement
    learning for traffic management. By doing small traffic simulations with
    multiple agents representing vehicles, the project seeks to analyze their
    interactions from a game-theoretic perspective. Agents learn and adapt their
    driving strategy through repeated interactions with other agents and the
    environment. The focus will be on developing small-scale traffic systems using
    multi-agent reinforcement learning and analyzing the resulting dynamics.
  ],
  authors: ((
    name: "Benedict Armstrong",
    department: [],
    organization: [ETH Zurich],
    location: [Zürich, Switzerland],
    email: "benedict.armstrong@inf.ethz.ch",
  ), (
    name: "Luis Wirth",
    department: [],
    organization: [ETH Zurich],
    location: [Zürich, Switzerland],
    email: "luwirth@ethz.ch",
  ), (
    name: "Felicia Scharitzer",
    department: [],
    organization: [ETH Zurich],
    location: [Zürich, Switzerland],
    email: "fscharitzer@student.ethz.ch",
  ), (
    name: "Noah Gigler",
    department: [],
    organization: [ETH Zurich],
    location: [Zürich, Switzerland],
    email: "ngigler@student.ethz.ch",
  ), (
    name: "Ankush Majmudar",
    department: [],
    organization: [ETH Zurich],
    location: [Zürich, Switzerland],
    email: "amajmudar@student.ethz.ch",
  ),),
  index-terms: (
    "multi-agent reinforcement learning",
    "traffic simulation",
    "game theory",
  ),
  bibliography: bibliography("refs.bib"),
)

= Introduction

Traffic bla bla ...

#figure(
  image("../src/flow_animation/output.gif"),
  caption: "Traffic simulation",
)