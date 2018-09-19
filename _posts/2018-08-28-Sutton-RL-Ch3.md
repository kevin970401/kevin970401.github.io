---
layout: post
title: "Sutton RL Ch.3 - Finite Markov Decision Process"
categories: RL
author: lee gunjun
---

# Ch.3 - Finite Markov Decision Process

----

MDP에서 action은 immediate reward에만 영향을 주지 않고 state가 바뀜으로써 future reward까지 영향을 준다.

Bandit problem에서 우리는 action value를 $q_\ast (a)$, action에 대한 함수로만 풀었지만 MDP에선 $q_\ast (s, a)$ 즉 action과 함께 state도 고려해야한다. 또는 우리는 해당 state의 optimal action selection의 action value를 나타내는 $v_\ast(s)$를 구하게 될 것이다.

## 3.1 - The Agent-Environment Interface

----

**agent**: learner or decision maker

**environment**: the thing agent interacts with, comprising everything outside the agent

agent가 action을 취하면 environment는 새로운 situation와 reward을 agent에게 준다. 

agent와 env는 discrete time steps에 interact한다. agent는 각 step마다 env의 state(=$S_t$)를 받는다. 이 state는 action(=$A_t$)들로부터 나온다. action을 하고 한 스텝이 지나면 agent는 reward(=$R_{t+1}$)를 받는다. 그리고 새로운 state($S_{t+1}$)를 얻는다.

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, \dots$$

위와 같은 *trajectory*를 얻을 것이다.

Finite MDP에서는 모든 states, actions and rewards의 집합들은 유한한 갯수의 원소들을 가진다. 그리고 $R_t$와 $S_t$는 이전 state와 action에 따르는 어떤 discrete한 확률 분포를 가진다.

$$p(s', r \vert s, a) = Pr\{S_t = s', R_t = r \vert S_{t-1} = s, A_{t-1} = a \}$$

$\vert$는 conditional prob을 나타냄. 위의 식으로 MDP의 dynamics를 정의한다.

당연히 $\sum_{s' \in S} \sum_{r \in R} p(s', r \vert s, a) = 1,\ for\ all\ s \in S, a \in A(s)$ 이다.

이 확률 p는 MDP를 완벽하게 나타낸다. $R_t$와 $S_t$는 단지 바로 전의 $S_{t-1}$와 $A_{t-1}$에만 영향을 받고 그 전의 state나 action들로 나타내어지지 않는다. 즉 state는 미래를 결정하는데 필요한 과거의 정보를 다 가지고 있어야 한다. 이런 성질을 *Markov property*라고 한다.

아까의 four-argument dynamics function을 변형 시켜보자

$$p(s' \vert s, a) = Pr\{S_t = s' \vert S_{t-1} = s, A_{t-1} = a \} = \sum_{r \in R} p(s', r \vert s, a ) $$

$$r(s, a) = \mathbb{E}[R_{t} \vert S_{t-1} = s, A_{t-1} = a] = \sum_{r \in R}r \sum_{s' \in S} p(s', r \vert s, a)$$

$$r(s, a, s') = \mathbb{E}[R_t \vert S_{t-1} = s, A_{t-1} = a, S_t = s'] = \sum_{r \in R}r \frac{p(s', r \vert s, a)}{p(s' \vert s, a)}$$

## 3.2 - Goals and Rewards

----

RL에서 agent의 목적은 곧 reward의 총합을 maximize 하는 것.

reward 를 설정할 땐 무엇을 원하는지에 초점이 맞춰져야 하지 어떻게 해야 한다가 아니다. (후자는 pseudo rewarding)

## 3.3 - Returns and Episodes

----

**Return** denoted $G_t$. $G_t = R_{t+1}+R_{t+2}+R_{t+3}+\dots+R_T$. where T is a final time step. final time step은 env와 agent의 interation이 끝나는 시점이고 이 때 한 *episode*가 종료됐다고 말한다. return episode 가 끝이 나야 위와 같이 정의할 수 있다. episode가 끝나지 않는 경우는 바로 밑에서 서술할 것이다. each episode 는 terminal state란 특별한 state 에 도달하면 끝남. 각 episode들은 서로 independent함. 이런 걸 episodic tasks라 한다. episodic tasks에서 우리는 non-terminal state를 $S$ terminal-state와 non-terminal state의 합집합을 $S^+$라 한다.

하지만 끝이 나지 않는 task도 있다. 이런걸 continuing tasks라 부른다. 이런 tasks에서는 $T = \infty$기 때문에 return 을 위처럼 정의하기 힘들다. time step마다 1 씩 얻는 task라 하면 return이 diverge할 수 있기 때문. 그래서 discounting이라는 새로운 개념을 추가한다. 우리의 목적은 discounted rewards의 합을 maximize하는 것이다.

$$G_t = R_{t+1}+\gamma R_{t+2} + \gamma ^2 R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma ^k R_{t+k+1}$$

where $\gamma$ is a parameter, $0 \le \gamma \le 1$ called the discount rate

이렇게 되면 time step마다 1씩 reward를 얻더라도 discounted rewards의 합은 수렴한다.

또한 연속적인 return은 아래와 같은 관계식을 가진다

$$\begin{matrix}
G_{t}&=&R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+2} + \dots \\
&=&R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+2} + \dots) \\
&=&R_{t+1} + \gamma G_{t+1}\\
\end{matrix}$$

## 3.4 - Unified Notation for Episodic and Continuing Tasks

----

앞에서 episodic tasks와 continuing tasks를 다뤘다. episodic tasks를 더 자세히 다루기 위해선 series of episodes를 고려해볼 필요가 있다. 우리는 이제 단순히 state를 time-step t에 대해서만 $S_t$라 할게 아니라 i-th episode인 것까지 고려하여 $S_{t,i}$를 고려해야 한다. 그런데 사실 굳이 episode까지 각각 구별해야 하는 일이 생기는 경우는 잘 없다. 그러니 그냥 $S_t$까지만 고려하자. (물론 $S_{t,i}$를 고려해야 하는 때가 온다.)

episidic tasks와 continuing tasks를 합칠 수도 있다. episodic tasks를 continuing tasks으로 만드는 것인데, episodic tasks의 마지막 terminal state를 absorbing state(transitions only to itself and that generates only rewards of zero)로 바꾸는 것이다. 

## 3.5 - Policies and Value Functions

----

거의 모든 rl algorithms들은 value functions을 예측한다. value function은 policy(어떻게 actions을 고를까)에 대해 정의된다.

policy은 한 state에서 가능한 actions들 중 어떤 action을 고르는 가에 대한 mapping이다. If the agent is following policy $\pi$ at time t, then $\pi(a \vert s)$ is the probability that $A_t = a$ if $S_t = s$. rl은 경험으로 부터 agent의 policy를 어떻게 바꿔야 return이 가장 클지 학습하는 것이다.

$$v_{\pi}(s) = \mathbb{E}[G_t \vert S_t = s] = \mathbb{E}_\pi \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t = s \right],\ for\ all\ s\in S$$

$v_{\pi}(s)$ is the **state-value function** for policy $\pi$

$$q_\pi (s, a) = \mathbb{E}_\pi \left[G_t \vert S_t = s, A_t = a \right] = \mathbb{E}_\pi \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t = s, A_t = a \right]$$

$q_{\pi}(s, a)$ is the **action-value function** for policy $\pi$

state-value function과 action-value function은 다음과 같은 관계식을 가진다.

$$v_\pi(s) = \sum_{a} \pi(a|s) q_\pi(s, a)$$

*Monte carlo methods*: 수 많은 random sample의 average로 value function을 estimate하는 방법

너무 많은 state가 있을 경우 각 state마다 value function을 저장하는 것은 practical 힘드므로 state 수 보다 적은 수의 parameter을 이용한다.(dqn이 그랬듯)

$$\begin{matrix}
v_\pi (s) & = & \mathbb{E}_\pi [G_t \vert S_t = s]\\
& = & \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \vert S_t = s]\\
& = & \sum_{a} \pi (a \vert s) \sum_{s'} \sum_{r} p(s', r \vert s, a) \left[ r + \gamma \mathbb{E}_\pi [G_{t+1} \vert S_{t+1} = s'] \right]\\
& = & \sum_{a} \pi (a \vert s) \sum_{s', r} p(s', r \vert s, a) \left[ r + \gamma v_\pi(s') \right]\ for\ all\ s \in S
\end{matrix}$$

위의 식을 **Bellman Equation** for $v_\pi$ 이라 한다. 이는 연속적인 두 state간의 value function의 관계를 알려준다.

*backup diagram*: rl 에 자주 등장하는 diagram. 그림은 생략

## 3.6 - Optimal Policies and Optimal Value Functions

----

두 policies $\pi, \pi'$를 생각해보자. 이 중 $\pi$의 expected return이 모든 state에 대해 $\pi'$의 expected return 보다 크면 $\pi \ge \pi'$라고 한다. 다시 말하면 $\pi \ge \pi'\ if\ and\ only\ if\ v_\pi(s) \ge v_{\pi'}(s)\ for\ all\ s \in S$

모든 policy보다 크거나 같은 policy가 꼭 하나씩 존재한다. 그걸 *optimal policy*라 한다. optimal policy는 하나 이상 존재할 수 있고 우리는 그걸 $\pi_\ast$라고 표기한다. 그 optimal policy들은 같은 state-value function을 공유하는 데 우리는 그걸 *optimal state-value function*이라 부른다.

$$v_\ast(s) = \max_\pi v_\pi (s)$$

또한 optimal policies는 같은 *optimal action-value function* 을 공유한다.

$$q_\ast(s, a) = \max_\pi q_\pi(s, a)$$

$v_\ast$와 $q_\ast$는 다음과 같은 관계식을 가진다.

$$q_\ast(s, a) = \mathbb{E}[R_{t+1}+\gamma v_\ast(S_{t+1}) \vert S_t = s, A_t = a]$$

optimal state-value function은 하나의 policy을 따르는 state-value function이기 때문에 당연히 위의 $v_\pi (s) = \sum_{a} \pi (a \vert s) \sum_{s', r} p(s', r \vert s, a) \left[ r + \gamma v_\pi(s') \right]\ for\ all\ s \in S$ 관계식을 만족한다. 하지만 optimal state-value function은 이보다 간단한 다른 식도 만족시킴을 보일 수 있다.

$$\begin{matrix}
v_\ast(s) & = & \max_{a \in A(s)} q_{\pi_\ast}(s, a)\\
& = & \max_{a} \mathbb{E}_{\pi_\ast} [G_t \vert S_t = s, A_t = a]\\
& = & \max_{a} \mathbb{E}_{\pi_\ast} [R_{t+1} + \gamma G_{t+1} \vert S_t = s, A_t = a]\\
& = & \max_{a} \mathbb{E} [R_{t+1} + \gamma v_\ast (S_{t+1}) \vert S_t = s, A_t = a]\\
& = & max_{a} \sum_{s', r} p(s', r \vert s, a) [r+\gamma v_\ast (s')]\\
\end{matrix}$$

위의 관계식을 *Bellman optimal equation*이라 부른다.

action-value function 의 *Bellman optimal equation*은 다음과 같다

$$\begin{matrix}
q_\ast(s, a) & = & \mathbb{E}[R_{t+1}+\gamma v_\ast(S_{t+1}) \vert S_t = s, A_t = a]\\
& = & \mathbb{E}[R_{t+1}+\gamma \max_{a'} q_\ast (S_{t+1}, a') \vert S_t = s, A_t = a]\\
& = & \sum_{s', r} p(s', r \vert s, a) [r+\gamma \max_{a'} q_\ast (s', a')]
\end{matrix}$$

$v_\ast$를 알고 있다면 optimal policy를 결정하는 것은 매우 쉽다. 각 state 마다 bellman optimality equation을 만족하는 a를 선택하면 된다. 이를 greedy라고 한다.

bellma optimality eq 를 통해 optimal policy를 찾는 것은 3가지 가정이 필요하다.

- dynamics of environment를 알고 있어야함
- 이를 풀 충분한 computational resources를 가지고 잇어야 함
- Markov property 

3개의 가정을 모두 충족할 만한 상황이 현실 문제에선 별로 안나와서 문제.

## 3.7 Optimality and Approximation

----

task가 작으면 각 state마다 action-value를 적은 array나 list등의 tabular를 만들어서 풀 수 있다. 이런 case를 tabular case라 하고 이렇게 푸는 것을 tabular methods라 한다. 그러나 실제론 tabular를 쓰기엔 메모리가 너무 많이 들기 때문에 적은 parameters들을 이용하여 action-value를 대략적으로 구한다.

reinforcement learning은 더 많이 만나는 state의 value function을 학습하는 데 더 많은 노력을 들인다. 이 점이 MDP를 approximately 푸는 다른 접근 방법들과의 차이점이다.
