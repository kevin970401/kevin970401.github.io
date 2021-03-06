---
layout: post
title: "Sutton RL Ch.1 - Markov Decision Process"
categories: RL
author: lee gunjun
---

# 1. MDP
----
## Definition - Markov
----
> A state $S_t$ is Markov if and only if <br>
> $\mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_1, ..., S_t]$

$$P_{ss'}=\mathbb{P}[S_{t+1} = s' | S_{t} = s]$$

$$P = \begin{bmatrix}
P_{11} & \cdots & P_{1n} \\
\vdots & \ddots & \vdots \\
P_{n1} & \cdots & P_{nn}
\end{bmatrix}$$

where each row of the matrix sums to 1

## Definition - Markov Process
----
> A Markov Process (or Markov Chain) is a type <S, P>
> - S is a set of states
> - P is a state transition probability matrix <br>
> $P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s]$

## Definition - Markov Reward Process
----
> Markov Reward Process is a tuple <S, P, R, $\gamma$>
> - S is a finite set of states
> - P is a state transition probability matrix
> $P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_{t} = s]$ <br>
> - R is a reward function, $R_s = \mathbb{E}[R_{t+1} | S_t = s]$ <br>
> - $\gamma$ is a discount factor, $\gamma \in [0, 1]$

## Definition - $G_t$ (discounted reward)
----
> The return $G_t$ is the total discounted reward from time-step t. <br>
> $G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma ^k R_{t+k+1}$

## Definition - Value function $v(s)$ (MRP)
----
> The state value function v(s) of an MRP is the expected return starting from state s <br>
> $v(s) = \mathbb{E} [G_t | S_t = s]$

The value function can be decomposed into two parts
- immediate reward $R_{t+1}$
- discounted value of successor state $\gamma v(S_{t+1})$

$$\begin{matrix}
v(s) &=&  \mathbb{E}[G_t | S_t = s]\\

&=& \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R{t+3} + \cdots | S_t = s] \\

&=& \mathbb{E}[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) | S_t = s] \\

&=& \mathbb{E}[R_{t+1} + \gamma G(t+1) | S_t = s] \\

&=& \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]
\end{matrix}$$

**Bellman Equation** in Matrix

$$v(s) = R_s + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}v(s')$$

if can be solved directly:

$$\begin{matrix}
v &=& R + \gamma Pv \\

(1 - \gamma P)v &=& R \\

v &=& (1 - \gamma P)^{-1} R
\end{matrix}$$

## Definition - Markov Decision Process
----
> A Markov Decision Process (MDP) is a tuple <S, A, P, R, $\gamma$>
> - S is a finite set of states
> - A is a finite set of actions
> - P is a state transition probability matrix,
> $P_{ss'}^a = \mathbb{P}[S_{t+1} = s' | S_t = s, A_t = a]$<br>
> R is a reward function, $R_s^a = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]$<br>
> $\gamma$ is a discount factor $\gamma \in [0, 1]$

## Definition - Policy $\pi$
----
> A policy $\pi$ is a distribution over actions given states,<br>
> $\pi (a|s) = \mathbb{P} [A_t = a | S_t = s]$

$$P_{s,s'}^{\pi} = \sum_{a \in A} \pi (a|s) P_{ss'}^a$$

$$R_s^{\pi} = \sum_{a \in A} \pi (a|s) R_s^a$$

$$\sum_{s' \in S} \sum_{a \in A} \pi (a|s) P_{ss'}^a = 1$$

## Definitaion - Value function $v_\pi (s)$ (MDP)
----
> The state-value function $v_\pi (s)$ of an MDP is the expected return<br>
> starting from state s, and the following policy $\pi$ <br>
> $v_\pi (s) = \mathbb{E}_\pi [G_t | S_t = s]$

## Definition - Value function $q_\pi (s, a)$ (MDP)
----
> The action-value function $q_\pi (s, a)$ is the expected return <br>
> starting from state s, taking action a, and the following policy $\pi$ <br>
> $q_\pi (s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]$

**Bellman Expectation Equation** at action-value function $q_\pi (s, a)$

$$q_\pi (s, a) = \mathbb{E}_\pi [R_{t+1} + \gamma q_\pi (S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

$$v_\pi (s) = \sum_{a \in A} \pi(a|s)q_{\pi} (s, a)$$

$$q_{\pi} (s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi (s')$$

$$v_\pi (s) = \sum_{a \in A} \pi(a|s)(\mathcal{R}_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi (s'))$$

$$q_{\pi} (s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} P_{ss'}^a (\sum_{a \in A} \pi(a|s)q_{\pi} (s, a))$$

$$v_\pi = R^\pi + \gamma P^\pi v_\pi$$

$$v_\pi = (1 - \gamma P^\pi)^{-1} R^\pi$$

## Optimal Value Function
----
> - The optimal state-value function $v_* (s)$ is the maximum value function over all policies <br>
> $v_* (s) = \max_\pi v_\pi (s)$ <br>
> - The optimal action-value function $q_* (s, a)$ is the maximum action-value function over all policies
> $q_* (s, a) = max_\pi q_\pi (s, a)$

An MDP is **solved** when we know the optimal value function

$\pi \ge \pi^{'}$ means $v_{\pi} (s)\ge v_{\pi^{'}}(s),\forall s$

### Theorem
----
> For any MDP
> - There exists an optimal policy $\pi_*$ that is better than or equal to all other policies, $\pi_x \ge \pi, \forall \pi$
> - All optimal policies achieve the optimal value function, $v_{\pi_{\ast}}(s)=v_\ast (s)$
> - All optimal policies achieve the optimal action-value function, $q_{\pi_{\ast}}(s, a)=q_\ast (s, a)$

An optimal policy can be found by maximising over $q_* (s, a)$<br>

$$
\pi_* (a|s)=
\begin{cases}
0, & if a = arg\max_{a \in A} q_*(s, a)\\
1, & otherwise\\
\end{cases}$$

- **There is always a deterministic optimal policy for any MDP**
- if we know $q_* (s, a)$, we immediately have the optimal policy

 **Bellman Optimality Equation for $v_*$**

$$v_* (s) = \max_{a} q_{\pi} (s, a)$$

$$q_{*} (s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_* (s')$$

$$v_* (s) = \max_{a} (\mathcal{R}_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_* (s'))$$

$$q_{*} (s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in S} (P_{ss'}^a \max_{a} q_{\pi} (s, a))$$
