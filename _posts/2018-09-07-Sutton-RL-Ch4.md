---
layout: post
title: "Sutton RL Ch.4 - Dynamic Programming"
categories: RL
author: lee gunjun
---

# Ch.4 - Dynamic Programming

----

**Dynamic Programming**은 env의 MDP와 같은 perfect model이 주어졌을 때 optimal policies를 구하는 algorithm들을 통칭한다.

classical DP algorithm은 사용하기엔 두 개의 큰 제한이 걸린다
- env의 perfect model이 주어져야 한다는 점
- 거대한 연산량

하지만 이런 제한에도 불구하고 theoretically 매우 중요한 개념이다.

이 장을 시작하기 앞어 env가 finite MDP(S, A, R이 모두 finite)라는 가정을 하겠다.

그러면 env의 dynamics는 $p(s', r \vert s, a)\ for\ all\ s \in S,\ a \in A(s),\ r \in R,\ and\ s' \in S^{+}$ 로 주어진다.

DP은 continuous state and action spaces에도 사용할 수 있지만 exact solutions은 오직 특별한 case에만 가능하다. 그래서 보통 continuous states and actions의 approximative solution을 구하기 위해서는 states와 actions quantization 하는 방법이 주로 쓰인다.

DP 의 키포인트는 value function을 이용하여 good policies를 찾는 것이다.

## 4.1 - Policy Evalution (Prediction)

----

먼저 policy $\pi$에 대한 state value function $v_\pi$ 을 estimate 하는 방법에 대해 생각해보자.

우리는 위를 **policy evaluation** 이라고 부른다. **prediction problem** 이라고도 부른다.

ch3에서 우리는 

$$\begin{matrix}\\
v_{\pi} (s) & = & \mathbb{E}_\pi [G_t \vert S_t = s] \\
& = & \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
& = & \mathbb{E}_\pi [R_{t+1} + \gamma v_{\pi} (S_{t+1}) \vert S_t = s]\\
& = & \sum_{a} \pi(a \vert s) \sum_{s', r} p(s', r \vert s, a) (r + \gamma v_{\pi}(s')) \\
\end{matrix}$$

$\gamma$가 1보다 작거나 모든 state가 policy를 따르다보면 terminate state에 도달할 수 있다고 하면 *existence and uniqueness of* $v_\pi$는 보장된다.

MDP의 state가 n개라면 n개의 위의 관계식을 얻을 수 있다.

우리는 이를 이용하여 MDP를 DP로 풀 수 있다.

맨처음 $v_0$는 모든 state s에 대해 $v_0(s) = 0$으로 두고

$$\begin{matrix}\\
v_{k+1}(s) & = & \mathbb{E}_{\pi} [R_{t+1} + \gamma v_k(S_{t+1}) \vert S_t = s]\\
& = & \sum_a \pi (a \vert s) \sum_{s', r} p(s', r \vert s, a) [r + \gamma v_k(s')]\\
\end{matrix}$$

을 통해 iteratively update한다. Clearly $v_k = v_\pi$일 때가 update가 중단 될 때이다.

이 algorithm을 **iterative policy evaluation**이라 한다.

policy를 update하기 위해 모든 successor states of s와 expected immediate rewards(=r)를 고려해야 하기 때문에 우리는 이를 **expected update**라 부른다. (하나의 어떤 successor state sample을 가지고 update하는 것이 아니라 모든 successor states의 기댓값을 가지고 하기 때문인듯)

DP algorithm에서 이뤄지는 모든 update들은 expected update들이다.

이를 구현하는 방법에는 두 가지가 있다.
- old value와 new value를 각각 저장하는 두개의 array을 이용하는 방법
- value를 저장하는 하나의 array만을 이용하여 *in place*로 update하는 방법

in place또한 $v_\pi$로 converge할 뿐더러 보통 in place가 converge 하는 속도가 두개의 array를 쓰는 방법보다 빠르다.

## 4.2 - Policy Improvement

----

Policy Evaluation을 통해 찾은 value function을 통해 우리는 이전 policy보다 좋은 policy를 찾을 수 있다.

action value function을 통해 더 좋은 a를 찾아보자.

$$\begin{matrix}
q_\pi(s, a) & = & \mathbb{E} [R_{t+1} + \gamma v_\pi(S_{t+1}) \vert S_t = s, A_t = a]\\
& = & \sum_{s', r} p(s', r \vert s, a) (r + \gamma v_\pi (s'))\\
\end{matrix}$$

언제나 $q_\pi (s, \pi'(s)) \ge v_\pi(s)\ for\ all\ s \in S$인 deterministic policy $\pi'(s)$을 찾을 수 있다.

위의 policy $\pi'$는 $v_{\pi'}(s) \ge v_\pi(s)$을 만족한다.

if there is strict inequality of $q_\pi (s, \pi'(s)) \ge v_\pi(s)$ at any state, then there must be strict inequality of $v_{\pi'}(s) \ge v_\pi(s)$ at least one state.

다시 말하면 모든 state에서 $q_\pi (s, \pi'(s)) \gt v_\pi(s)$이면 $v_{\pi'}(s) \gt v_\pi(s)$ 이다.

이는 greedy로 찾아낼 수 있다.

$$\begin{matrix}
\pi'(s) & = & arg \max_a q_\pi (s, a)\\
& = & arg \max_a \mathbb{E} [R_{t+1} + \gamma v_\pi (S_{t+1}) \vert S_t = s, A_t = a]\\
& = & arg \max_a \sum_{s', r} p(s', r \vert s, a) [r + \gamma v_\pi (s')]\\
\end{matrix}$$

이를 **policy improvement**라 한다.

만약 greedy policy $\pi'$이 old policy $\pi$와 같다고 해보자. 즉 $v_\pi = v_{\pi'}$ 이면

$$\begin{matrix}
v_{\pi'}(s) & = & \max_a \mathbb{E} [R_{t+1} + \gamma v_{\pi'}(S_{t+1}) \vert S_t = s, A_t = a]\\
& = & \max_a \sum_{s', r} p(s', r \vert s, a) [r+ \gamma v_{\pi'} (s')]
\end{matrix}$$

가 되는데 이는 Bellman optimality equation을 만족한다. 즉 $v_{\pi'}$는 optimal policy이다.

policy improvement는 이미 optimal policy인 경우를 제외하고 **항상 strictly better policy**을 준다. 

## 4.3 - Policy Iteration

----

첫 policy $\pi$를 policy evaluation을 해서 $v_\pi$를 구하고 이를 이용하여 policy improvement을 해서 $\pi'$를 구하고 이를 policy evaluation을 해서 $v_{\pi'}$를 구하고... 이를 반복하여 optimal policy를 구할 수 있다. 이를 **policy iteration** 이라 부른다.

policy improvement를 할 때마다 strictly 좋은 policy를 구할 수 있고(이미 optimal policy 인 경우 제외) 이는 optimal policy로 converge한다.

policy iteration은 첫 policy가 어떻든 결국 optimal policy에 도달할 수 있다.

## 4.4 - Value Iteration

----

policy iteration의 문제점은 매 iteration마다 computational cost가 매우 높은 policy evaluation이 필요하다는 것이다. 그래서 이런 고민을 해본다. 꼭 exact value function을 구해야만 하는가? 

사실은 꼭 exact value function에 converge 할 때까지 policy evaluation을 진행해야 하는 것이 아니다. 그 예로 모든 state에 대해 한번씩만 특별한 방법으로 value function을 update하고 난 뒤에 멈춰도 된다. 이 algorithm을 **value iteration**이라 부른다. *심지어 이 방법은 policy를 이용하지 않는다!*

### Value iteration

$$\begin{matrix}
v_{k+1} (s) & = & \max_a \mathbb{E} [R_{t+1} + \gamma v_k(S_{t+1}) \vert S_t = s, A_t = a]\\
& = & \max_a \sum_{s', r} p(s', r \vert s, a) [r + \gamma v_k (s')] \\
\end{matrix}$$

를 모든 state s에 대해 계산한 후 종료한다.

아무 initial value function으로 시작하더라도 sequence $\{v_k\}$는 optimal value function $v_\star$에 converge한다.

이 value iteration은 사실 policy evaluation을 한번만(여기서 한번만은 policy evaluation의 전체 state를 도는 loop문을 한번만 돈다는 것) 하고 policy improvement를 한번 하는 것과 같다.

이를 변형하여 policy evaluation을 두번하고 policy improvement를 한번 하는 것으로 더 빠른 converge를 할 수 도 있다. (더 느릴 수도 있다.)

## 4.5 - Asynchronous Dynamic Programming

----

DP의 단점은 모든 state에 대해 계산을 해야 한다는 것이다. state set이 크면 큰일남. 오목, back gammon, 바둑 같은 거 절대 못풂.

Asynchronous DP algorithm은 priority를 주는 느낌. 다른 state가 한두번 update 될 때 어떤 state는 여러번 update될 수 있음. 하지만 역시 converge하기 위해선 모든 state를 update 하는 것을 반복해야 한다.

## 4.6 - Generalized Policy Iteration

----

Policy iteration에서 Policy evaluation과 Policy improvement는 번갈아 가면서 일어났다. 하지만 꼭 이렇게 Policy evaluation이 끝나야 Policy improvement를 하고 Policy improvement가 끝나야 Policy evaluation을 해야만 하는 건 아니다. 

예시로 value iteration에서 우리는 policy evaluation이 끝나기 전에 Policy improvement를 했었다.

Asynchronous Dynamic Programming에서도 policy evaluation이 끝나기 전에 Policy improvement를 했었다

우리는 policy evaluation과 policy improvement가 interact하는 것을 **Generalized policy iteration**라고 부른다. 거의 모든 rl methods는 GPI로 설명되어 진다. 즉 value function은 policy 를 토대로 improve되고 policy는 value function을 토대로 improve 되는 것이다. (policy $\pi$에서 $v_\pi$를, $v_\pi$에서 $\pi$를 구함)

## 4.7 - Efficiency of Dynamic Programming

----

DP는 웬만해선 practical하지 않다. 하지만 썩 그렇게 나쁜 것만도 아닌게 시간 복잡도를 계산해보면 state와 action의 수에 polynomial하다.

deterministic policies 갯수가 $k^n$개 (k: action 수, n: state 수)므로

DP 는 *curse of dimensionality* 문제점을 가지고 있다. 
