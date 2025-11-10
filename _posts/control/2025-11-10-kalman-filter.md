---
layout: default
title: "Kalman filter"
date: 2025-11-10 00:09:00 +0900
categories: control_system
permalink: /20250829/2025-09-15-drone-simulation.html
---

# Kalman filter

## What is baysian filter?

To know kalman filter, we need to learn baysian filter first.

### Baysian Theory

$$


$$

## What is Kalman filter?

The Kalman Filter assumes a linear system with Gaussian noise.

### State Transition Model

$$
x_k = Fx_{k-1} + w_k
$$

-   $x_k$: State at time $k$
-   $F$: State transition matrix
-   $w_k$: Process noise, Gaussian with mean $0$ and convariance $Q(w_k ~ N(0, Q))$, representing model uncertainty.
-
