---
layout: post
title: "Sutton RL Ch.2 - Multi-armed bandits "
categories: RL
author: lee gunjun
---

# Ch.2 - Multi-armed bandits

----

## 2.1 - A k-armed Bandit Problem

----

k-armed bandits
- bandit은 슬롯머신을 말하고 arm은 그 슬롯머신의 레버를 뜻함. k개의 슬롯머신이 있다는 의미.
- k개의 action이 있고 각 action마다 정적 확률 분포를 가지는 reward가 numerical하게 주어짐.
- 우리의 목표는 expected total reward over some time period를 maximize 하는 것.
- time step t에 고른 action을 $A_t$, 그에 따른 reward를 $R_t$라 하자.
- **action value**: $q_* (a)$ 는 $\mathbb{E} [R_t\vert A_t = a]$ 로 정의된다.
  - a라는 action을 택했을 때 reward의 expectation임
  - action value를 안다면 k-armed bandits를 깬것임. 가장 높은 action value를 가지는 action을 선택하면 가장 큰 reward를 얻을 것이므로.
- action value를 확실하게 알지 못하므로 **estimated action value** at time step t 를 $Q_t(a)$ 라고 둔다.
  - 이 $Q_t(a)$를 최대한 $q_\ast (a)$에 근접하게 만드는 것이 목표이다.
- 매 time step마다 *estimated action value*가 가장 큰 action이 하나 이상 존재할 것이다. 우리는 이를 **greedy actions**이라 부른다.
- 이 *greedy actions* 중 하나를 고르는 것을 **exploiting**이라 하고 nongreedy actions 중 하나를 고르는 것을 **exploring**이라 한다.
  - non greedy action 중 action value가 현재 greedy action보다 큰 경우가 있을 수 있으니, 이를 찾기 위해 *exploring*이 필요하다.
- **certainty**는 estimated action value가 action value와 같을 때를 말한다.
- *exploiting*과 *exploring*의 밸런스를 맞추는 것 또한 rl 에서의 중요한 문제.

## 2.2 - Action-value Methods

----

action value를 구하는 가장 방법 중 하나는 실제로 그 action을 택했을 때 받은 reward들의 평균을 구하는 것이다.(=sample average method)

**sample average method**

$$Q_t (a) = \frac{sum\ of\ rewards\ when\ a\ taken\ prior\ to\ t}{numver\ of\ times\ a\ taken\ prior\ to\ t} = \frac{\sum_{i=1}^{t-1} R_i * 1_{A_i = a}}{\sum_{i=1}^{t-1} 1_{A_i = a}}$$

where $1_{predicate}$ denotes the random variable that is 1 if predicate is true and 0 if is is not.

*greedy action selection*은 $A_t = arg \max_{a} Q_t (a)$ 인 $A_t$를 고르는 것이다.

$\epsilon -greedy\ action\ selection$은 $\epsilon$ 의 확률로 greedy action을 고르고 나머지 확률로 nongreedy actions중 하나를 고르는 것이다.

## 2.3 - The 10-armed Testbed

----

실험 해 보면 실제로 $\epsilon -greedy$가 greedy보다 결과가 좋게 나온다.

## 2.4 - Incremental Implementation (skip 가능)

----

우리는 아까 estimated action value을 구할 때 observed rewards 들의 평균을 계산했었다. 이걸 어떻게 컴퓨터에서 효율적인 방법으로 구현할 수 있을까

제한 조건: constant memory, constant per-time-step computation.

$$ Q_{n} = \frac{R_1 + R_2 + \dots + R_n}{n-1} $$

이를 변형 시켜보자.

$$ \begin{matrix}
Q_{n+1}&=& \frac{1}{n} \sum_{i=1}^{n} R_i \\
&=& \frac{1}{n} \left(R_n + \sum_{i=1}^{n-1} R_i\right)\\
&=& \frac{1}{n} \left(R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i\right) \\
&=& \frac{1}{n} \left(R_n + (n-1)Q_n \right) \\
&=& \frac{1}{n} \left(R_n + nQ_n - Q_n \right)\\
&=& Q_n + \frac{1}{n} [R_n - Q_n] \\
\end{matrix} $$

즉 정리하면

$$ \begin{matrix}\therefore 
Q_{n+1}&=&Q_n + \frac{1}{n} [R_n - Q_n] \\
&=&[1-\frac{1}{n}]Q_n + \frac{1}{n} R_n
\end{matrix}$$

이렇게 함으로써 constant memory, constant per-time-step computation의 제한 조건에서 estimated action value를 구할 수 있다.

*개인의견*: 위 처럼 하면 floating point 때문에 오차 계속 커짐. 그냥 cumulative reward 만 계속 update 하고 시행햇수로 나눠주자.

## 2.5 - Tracking a Nonstationary Problem

----

stationary bandit problem에서는 reward probability가 시간이 지나도 변하지 않기 때문에 observed rewards의 average를 구하는 것으로 충분했다.

실제로 rl 할때는 nonstationary problem(reward probability가 시간이 지남에 따라 변함)을 자주 만나기 때문에 이에 대한 대처가 필요하다.

가장 유명한 방법은 constant step-size parameter를 이용하는 것이다.

$$Q_{n+1} = Q_{n} + \alpha [R_n - Q_n]\ \ where \ \alpha\in(0,1]$$

또는 (같은 식이다)

$$Q_{n+1} = [1-\alpha]Q_{n} + \alpha R_n\ \ where \ \alpha\in(0,1]$$

$Q_n$은 weighted average of past rewards 이다.

$\alpha$를 n-th selection of action a 의 함수 $\alpha_n(a)$로 나타낼 수 있는데 $\alpha_n(a)=\frac{1}{n}$ 이면 sample average method가 된다. sample average method is guaranteed to converge to the true action values by the law of large numbers. but of course convergence is not guaranteed for all choices of the sequence {$\alpha_n (a)$}.

converge 하기 위해선 다음 과 같은 두 조건이 필요하다

$$\sum_{n=1}^{\infty}\alpha_n(a) = \infty \ \ and \ \ \sum_{n=1}^{\infty} \alpha_n^2(a)<\infty$$

첫번째 조건은 초기값이 어떻든 간에 상관없게 하려는 조건이고

두번째 조건은 그러면서도 convergence하게 하기 위한 조건이라고 직관적으로 생각하자.

그런데 constant step-size parameter를 이용하는 건 위의 조건을 만족하지 않는데 왜 쓰는 것인지 의문이 들 것이다. 이에 대해선 두가지 답이 있다.

1. nonstationary에서는 최근의 reward가 중요하기 때문
2. nonstationary에서 저 두 조건을 만족하는 것을 쓰면 converge가 너무너무 오래걸린다.

그래서 저 두 조건은 거의 이론에만 등장하고 실제론 잘 안 쓰인다.

## 2.6 - Optimistic Initial Values

----

지금까지 말한 것들은 전부 initial action-value estimates $Q_1(a)$에 영향을 받는다. 이걸 initial estimates에 **biased** 되어있다고 한다. average methods에서 constant step-size parameter를 이용하면 bias는 영원히 존재할 것이다.

근데 practice에선 usually 별문제가 안되고 오히려 prior knowledge를 전달할 수 있어서 도움이 되기도 한다. optimistic한 초기값을 줌으로써 exploration을 고도화 시키는 방법이 될 수도 있다. 
  - 예를 들어 k-armed bandit machine에서 각 machine들의 expected reward가 1밖에 되지 않는데 초기값을 전부 5로 줬다고 하면 exploration이 더 잘 될것이다.
  - 이러한 테크닉을 *optimistic initial values* 라고 한다.

## 2.7 - Upper-Confidence-Bound Action Selection

----

exploration은 action-value estimates의 uncertainty 때문에 필요하다.

우리는 지금까지 exploration을 위한 sampling 방법으로 $\epsilon$-greedy를 사용했는데 $\epsilon$-greedy는 non-greedy actions중에 어떤 preference가 없다는 단점이 있다(예를 들어 non-greedy 중 estimated action-value가 크면 좀 더 높은 확률을 준다 거나 하는 것들). non-greedy 중에 optimal이 될 확률이 높은 것을 더 많이 sample하는 게 있다면 좋을 것이다. 그래서 우리는 Upeer-Confidence-Bound action selection(UCB)를 만들었다.

$$A_t = arg\max_a \left(Q_t(a) + c\sqrt{\frac{ln(t)}{N_t(a)}}\right)$$

$N_t(a)$는 시간 t 까지 action a 가 시행된 횟수이다. 많이 뽑힌 action이면 뒤의 루트 씌어진 항이 작아지고 적게 뽑힌 action이면 뒤의 항의 클것이다. c는 얼마나 uncertainty에 비중을 둘지 조절하는 상수다.

이렇게 하면 어떤 action-value의 uncertainty(or variance) 까지 고려하여 sampling 할 수 있게 된다

## 2.8 - Gradient Bandit Algorithms

----

지금까지 우리는 action-value을 estimate하고 estimates를 이용하여 action을 골랐다. 그런데 action을 고르는 방법엔 이렇게 action-value를 estimate하는 방법만 있는 것이 아니다.

지금 section에서는 다른 방법으로 action을 고르는 방법을 소개할 것이다.

$H_t(a)$ is a numerical preference for each action a. soft-max distribution을 이용하여 action probabilities를 구할 수 있다.

$$Pr\{A_t = a\} = \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} = \pi_t(a)$$

새로운 notation $\pi_t(a)$가 나왔는데 이는 time t에서 action a를 선택할 확률이다. 앞으로 자주 쓸 notation이다.

initially all preferences are the same(e.g., $H_1(a) = 0$, for all a)

there is a natural learning algorithm for this setting based on the idea of stochastic gradient ascent. On each step, after selecting action $A_t$ and receiving the reward $R_t$, preferneces are updated by

$$\begin{matrix}
H_{t+1}(A_t)&=&H_t(A_t) + \alpha (R_t-\bar{R}_t)(1-\pi_t(A_t))& and\\
H_{t+1}(a)&=&H_t(a) - \alpha (R_t-\bar{R}_t)\pi_t(a)& for\ all\ a \neq A\\
\end{matrix}$$

where $\alpha > 0$ is a step-size parameter, and $\bar{R}_t$ 은 지금까지 얻은 모든 액션들에 대한 rewards들의 총합이다. $\bar{R}_t$가 지금 reward가 좋은 건지 나쁜건지 판단해주는 baseline이 되어준다.

## 2.9 - Associative Search (Contextual Bandits)

----

우리는 지금까지 nonassociative tasks만 다뤘다. nonassociative tasks는 상황(state)가 다르다고 해서 취할 수 있는 action의 종류가 다르지 않은 task를 말하고 반대로 associative tasks는 상황마다 취할 수 있는 action의 종류가 다를 수 있는 task를 말한다. multi-armed bandits 에서는 state가 하나만 있었으므로 당연히 nonassociative task였다. 그런데 일반적인 reinforcement learning task에는 하나 이상의 상황이 있다. nonassociative task을 associative task로 확장시키는 것을 해보자

**Contextual Bandit**: k-armed bandit task가 여러개 있다고 가정하자. 매 스텝마다 tasks들 중 하나를 random하게 고르게 된다. 각 task마다 고유의 color들로 슬롯 머신이 색칠되어 있다면 우리는 스텝마다 우리가 어떤 상황인지 구별할수 있다. 각 task마다 슬롯머신의 갯수는 바뀔 수 있다.

위와 같은 상황이 바로 *associative search task*이다. 이를 해결하기 위해선 state와 연관된 action들 중 제일 좋은 것을 골라야 할것이다. 위와 같은 문제를 그저 nonstationary task 로 보고 풀려하면 잘 안풀릴것이다. 우리는 다음 챕터 부터 이런 Associative search와 관련된 문제를 고려할 것이다.
