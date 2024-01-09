This repo holds code and data to perform experiments with Nick's "Geometric Robustness Metric for Machine Learning Models".

## Executive Summary

An intuitive theoretical method to test the stability and explainability of a model in production in
real-time. Some efforts have already indicated the effectiveness of this method, so what remains to do
is diligently produce a scientific-level work.

The particular proposal is to measure the “harmoniticity" of the model, specifically the degree to
which the model’s decision function f satisfies the harmonic property,

$$
\nabla^2 f = 0
$$

This can be approximated by defining a 'ball' around a test point, measuring the model on the ball, taking its average, and comparing somehow to the central point's model value.


