---
layout: default
title: "Kalman filter"
date: 2025-11-10 00:09:00 +0900
categories: control_system
permalink: /20250829/2025-11-10-kalman-filter.html
---

# Kalman filter

Before learning Kalman filter, it's essential to understand Bayes' theorem, as the Kalman filter is fundamentally based on it.

## Table of Contents

-   [Kalman filter](#kalman-filter)

## Bayes' Theorem

The [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) is an approach to statistical inference, where it is used to invert the probability of observations given a model configuration.

### Statement of theorem

Bayes' theorem is stated mathmatically as the following equation:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

where $A$ and $B$ are events and $P(B) \neq 0$.

-   $P(A\|B)$ is a [conditional probability](https://en.wikipedia.org/wiki/Conditional_probability); the probability of event $A$ occurring given that $B$ is true.
-   $P(B\|A)$ is also a conditional probability; the probability of event $B$ occurring given that $A$ is true.

### Proof

#### For events (Discrete)

$$
P(A|B) = \frac{P(A \cap B)}{P(B)},\ \text{if} \ P(B) \neq 0
$$

where $P(A \cap B)$ is the probability of both $A$ and $B$ being true. Similarly,

$$
P(B|A) = \frac{P(A \cap B)}{P(A)},\ \text{if} \ P(A) \neq 0
$$

Solving for $P(A \cap B)$ and substituting into the above expression for $P(A\|B)$

## Bayes' Filter

## What is Kalman filter?

The Kalman Filter assumes a linear system with Gaussian noise.

### State Transition Model

$$
x_k = Fx_{k-1} + w_k
$$

-   $x_k$: State at time $k$
-   $F$: State transition matrix
-   $w_k$: Process noise, Gaussian with mean $0$ and convariance $Q(w_k ~ N(0, Q))$, representing model uncertainty.
