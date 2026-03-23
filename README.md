# 🔷 Dropout in Neural Networks

## 📌 Overview

Dropout is a stochastic regularization technique used in neural networks to reduce overfitting. During training, it randomly disables a subset of neurons in each iteration, forcing the model to learn robust and redundant representations.

---

## 🧠 Definition

Dropout randomly sets neuron activations to zero with probability \( p \) during training.

- Drop probability: \( p \)
- Keep probability: \( q = 1 - p \)

---

## 🔍 Core Intuition

Instead of training a single network, dropout effectively trains an ensemble of many smaller subnetworks.

At each iteration:
- A random subset of neurons is removed
- The network cannot rely on specific neurons
- Encourages generalization and robustness

---

## ⚙️ Mathematical Formulation

Let:
- \( h \): activation vector
- \( r \sim \text{Bernoulli}(q) \): dropout mask

### Standard Dropout:
\[
\tilde{h} = r \odot h
\]

### Inverted Dropout (used in practice):
\[
\tilde{h} = \frac{r \odot h}{q}
\]

where \( \odot \) denotes element-wise multiplication.

---

## 🔄 Training Procedure

For each mini-batch:

1. Sample dropout mask:
   \[
   r_i \sim \text{Bernoulli}(q)
   \]

2. Apply mask:
   \[
   h_i^{drop} = \frac{r_i \cdot h_i}{q}
   \]

3. Forward pass continues

4. Backpropagation:
   - Gradients flow only through active neurons
   - Dropped neurons receive zero gradients

---

## 🔢 Numerical Example

Given:
- Activation: \( h = [2, 4, 6] \)
- Dropout probability: \( p = 0.5 \Rightarrow q = 0.5 \)

### Iteration 1:
Mask: \( r = [1, 0, 1] \)

\[
\tilde{h} = \frac{[2, 0, 6]}{0.5} = [4, 0, 12]
\]

### Iteration 2:
Mask: \( r = [0, 1, 0] \)

\[
\tilde{h} = \frac{[0, 4, 0]}{0.5} = [0, 8, 0]
\]

---

## 🚀 Inference Phase

During inference:
- Dropout is disabled
- Full network is used

\[
\tilde{h} = h
\]

Reason:
- Scaling is already handled during training (inverted dropout)

---

## 📊 Why Dropout Works

- Prevents co-adaptation of neurons
- Acts as implicit ensemble learning
- Improves generalization

---

## ⚠️ Common Pitfalls

- ❌ Using too high dropout → underfitting
- ❌ Applying dropout during inference
- ❌ Assuming same neurons drop every iteration
- ❌ Combining improperly with BatchNorm

---

## 🧩 Applications

- **MLPs**: After dense layers
- **CNNs**: Often after fully connected layers
- **Transformers**:
  - Attention layers
  - Feedforward blocks

---

## 📚 Reference

- Srivastava et al., 2014  
  *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*

---

## ✅ Key Takeaway

Dropout randomly samples a different subnetwork at every iteration, making the model more robust and reducing overfitting.
