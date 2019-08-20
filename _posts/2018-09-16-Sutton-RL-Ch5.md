---
layout: post
title: "Sutton RL Ch.5 - Monte Carlo Methods(작성 중)"
categories: RL
author: lee gunjun
---

# Ch.5 - Monte Carlo Methods

----

이 챕터에서 우리는 처음으로 환경에 대한 완전한 지식 없이 value function과 optimal policy를 찾아내는 learning methods를 알아 본다. 

**Monte Carlo Methods**는 optimal policy 를 찾기 위해서 오직 *experience* (실제 환경과 interaction하여 얻은 sample sequences of states, actions and rewards) 만이 필요하다. experience을 통해 learning하는 것은 환경의 complete knowledge없이도 optimal policy를 찾을 수 있게 해준다.

Monte Carlo Method는 sample returns을 평균내는 방식에 기반한 방법이다. 우리는 여기서 episodic task만 다룰 것이다. 한 episode가 끝날때 비로소 우리는 policy와 value estimates를 바꿀 것이다. 

nonstationary(state가 바뀜)을 다루기 위해 우리는 4장에서 다룬 GPI을 적용할 것이다. DP에서는 value functions을 **계산**했지만, 여기서는 value function을 sample들을 통해 **learn** 할 것이다.

## 5.1 - Monte Carlo Prediction

----

먼저 주어진 policy $\pi$에 대한 value function을 learning 하는 것에 대해 알아보자. 

가장 떠올리기 쉬운 방법은 averaging the returns observed after visits to that state. 많이 경험이 쌓일수록 실제 value에 converge 할 것이다. 이 개념은 모든 Monte Carlo Methods의 기반이 된다.

episode에서 state s가 일어난 걸 *visit*이라 한다. 물론 한 state를 한 episode에서 여러번 visit 할 수 있다. 처음 state s에 visit 한 걸 *first visit* to s라 하자

*first visit* MC method는 state s의 *first visit* 만을 이용하여 $v_\pi(s)$을 estimate한다. 이와 반대로 *every-visit* MC Method는 state s에 대한 모든 visit을 고려하여 value function을 estimate한다. 물론 두 방법 모두 그냥 return 들을 averaging 하는 것이다. **이 두 MC Methods는 비슷해 보이지만 서로 다른 theoretocal properties에 기반을 두고있다.**

*first visit* MC method가 이 챕터에서 다룰 내용이고 *every-visit* MC Method는 챕터 9와 12에서 다룰 것이다.

*first visit* MC method 가 converge 함은 쉽게 보일 수 있다. 각 return 은 indep 고, identically distributed estimate of $v_\pi(s)$. the law of large numbers 에 의해 $\frac{1}{\sqrt{n}}$ 의 속도로 converge 함.

*Every visit* MC 는 $\frac{1}{n^2}$ 로 converge 함.

env의 complete knowledge를 갖고 있어도 DP를 적용하기 힘든 경우가 있다. 먼저 DP는 probability를 전부 계산해서 알고 있어야 하고 이를 이용하여 value function과 optimal policy를 얻어낸다. 그런데 보통 보드게임에선 probability를 계산하는 것부터가 computation이 너무 많이 필요해서 사실적으로 불가능한 경우가 있다. 이런 경우 DP 대신 Monte Carlo를 쓰면 매우 간단하다. 그리고 이런 상황은 매우 빈번히 일어난다.

MC Method에서 중요한 점은 각각의 state에 대한 estimate들은 indep하다는 것이다. 어떤 state의 estimate는 다른 어떤 state의 estimate를 기반으로 만들어지지 않는다. 이게 DP하고 다른 점이다. 즉 **MC Methods don't bootstrap**

또한 MC Methods 에서 한 state의 value function 값을 알기 위해 다른 state 들의 value function 까지 알 필요가 없다는 것은 다시 말해 한 state의 value function 만을 (또는 전체 state 중 몇개의 state만) 알고싶은 상황에서 MC Method 를 쓰는 것은 매우 좋은 선택이라는 것이다. 관심있는 state에서 시작하는 episode들만 잔뜩 만들고 average 하면된다.

## 5.2 - Monte Carlo Estimation of Action Values

----

모델을 모른다면 state value function 보다 action value function 을 구하는 것이 더 좋을 것이다.

모델을 알고 있으면 state value function은 policy를 결정하는데 쓰일 수 있다. 그 다음 단계를 살펴보고 가장 좋은 action을 하면 되니까. 그런데 다음 단계를 살펴보기 위해서는 $p(s', r \vert s, a)$가 필요하다. 왜냐면 가장 좋은 action 을 고르기위해선 $\arg\max_a \sum_{s', r} p(s', r \vert s, a) (r + v_\pi(s'))$ 를 찾기 때문.

따라서 $v_\pi$ 만을 알아서는 최대의 expected return을 가지는 action을 알아낼 수 없고 모델의 knowledge까지 있어야 가능하다.

그런데 state value function이 아닌 action value function을 알고 있다고 하면 우리는 그저

$$arg\max_a q(s, a)$$

만을 계산하면 된다. 이 과정에서는 모델에 대한 knowledge가 필요하지 않다.

우리는 이제 MC Methods를 action value function q(s, a)에 적용한다.

*visit*은 다시 state-action pair로 정의된다. state s에서 action a 를 취한 것이 하나의 visit 이 된다. 이제 단순히 같은 state에 온다고 해서 같은 visit이 되는 것이 아니라 같은 state 더라도 다른 action을 한다면 다른 visit이 된다. 그래서 *first visit*과 *every visit*에도 변화가 생긴다.

*first visit*은 state s에서 action a를 처음으로 한 visit을 말한다.

아까 visit을 state 로만 다룰 때와 다르게 state-action pair를 다루는 지금은 방문하지 않는 state-action pairs 들이 상당히 많다. 만약 policy $\pi$ 가 deterministic 하다면 한 state에 대해 state action pair는 단 하나만 계속 만들어지고 다른 action들은 꿈도 못꾼다. 이건 굉장히 심각한 문제이다. 

이러한 문제를 **maintaining exploration** 이라고 한다. 간단한 해결 방법으로는 시작할 state-action pair를 정해주는데, 모델의 모든 state-action pair가 시작될 확률이 0이 아니게끔 하는 것이다. 이를 *exploring starts* 방법이라 한다.

exploring start 는 좋지만 대개 실제 상황에서는 써먹지 못하는데, 이렇게 인위적으로 starting을 정해줄 수 없는 env가 대부분이기 때문이다. 다른 방법으로는 policy가 어디서도 deterministic 하지 않고 stochastic 하게 해서 어떤 state-action pair도 each state에서 뽑힐 확률이 nonzero probabilty을 가지게끔 하는 것이다.

후에 위의 두 방법 모두 다룰 거지만 일단은 exploring starts 를 가정하자.

## 5.3 Monte Carlo Control

----

자 이제 MC를 이용해서 어떻게 control 할지 알아보자. control 한다는 것은 approximative optimal policy를 구한다는 것이다. 전반적으론 DP에서 다룬 GPI(Generalized Policy Iteration)과 같은 양상이다. 지금 policy 에 대한 value function 을 반복하여 근사하게 구하고, 그를 이용하여 policy를 구하고, 다시 value function을 가깝게 구하고... 반복

$$\pi_0 \rightarrow^{E} q_{\pi_0} \rightarrow^{I} \pi_1 \rightarrow^{E} \cdots \rightarrow^{I} \pi_\ast \rightarrow^{E} q_{\pi_\ast}$$

E는 policy evaluation 을, I 는 policy improvement 를 의미함.

Policy evaluation은 많은 episodes를 뽑아내서 action-value function을 가깝게 구하는 것인데, 일단 여기서 우리는 두 개의 가정을 하는데, 첫째로 exploring starts를 한다고 가정하고, 둘째로 무한번 sample해서 아주 정확한 action-value function을 구한다고 가정한다. 이런 가정 하에선 $\pi_k$ 의 action value function $q_{\pi_k}$ 를 정확하게 구해낼 수 있다.

Policy improvement 은 그렇게 구한 action-value function을 이용해서 policy 를 iterate 하는 것이다. 이때 우리는 greedy를 사용하여 deterministic 하게 action를 선택한다. action-value funtion 을 가지고 있기 때문에 model 에 대한 정보 없이 greedy 한 policy 를 잧을 수 있다.

$$\pi(s) = arg\max_a q(s, a)$$

앞에서 exploring starts를 가정했으므로 policy가 deterministic 해도 괜찮다.

이러면 DP 처럼 optimal policy 로 converge 한다.

위에서 우리는 MC method 의 convergence 를 위해 아주 무리한 가정 두가지를 했다. 첫째로 exploring starts를 가정했고, 둘째로 무한개의 episodes를 구한다는 것이었다. 실제로 MC Methods를 사용할때는 위의 두가지 가정을 할 수 있는 상황은 없다. 따라서 이 가정들을 없애야 한다. 첫번째 가정은 일단 남겨두고 두번째 가정을 없애보자.

finite episodes를 통해 action-value function을 approximately estimate 한다해도 optimal로 converge 할 수 있을까?

사실 이는 DP Methods에서도 대두됐던 문제이다. (결론은 converge 함이 증명되지 않았다..) DP와 MC 모두 이를 풀기 위한 두가지 방법이 존재한다. 

첫째로 충분히 value evaluation을 진행하는 것이다. 100% 정확한 value evaluation을 하기 위해서는 무한개의 episode가 필요하지만, 99.999% 정확한 value evaluation을 구하기 위해선 유한개의 episodes면 된다. 근데 이래도 너무 많은 episode 가 필요하다.

둘째로, policy improvement 하기 전에 policy evaluation 을 끝마치는 걸 포기하는 것이다. 말이 복잡한데 그냥 episode 조금 돌려서 policy evaluation 하고 이를 통해 policy improvement 한다는 것이다. episode 를 하나만 사용하는 것이 바로 전에 살펴봤던 value iteration 이다.

MC iteration 에서 episode-by-episode 로 evaluation과 improvement를 바꾸는 것은 자연스러운 것이다. 이와 exploring starts를 이용한 *Monte Carlo ES* (Monte Carlo with Exploring Starts) algorithm 를 소개한다.


### Monte Carlo ES(Exploring Starts), for estimating optimal policy

![](/assets/images/rl5/MCES.png)

MC ES의 주목할 점은 return이 어떤 policy에서 나온 것이든 상관하지 않고 다 accumulate 한 다음에 average를 하는 것으로 value function을 구한다는 점이다. 즉, MC ES는 어떤 suboptimal policy에 converge 하지 않는다. (suboptimal policy에 converge 한다는 것은 어떤 policy에 대해 value function을 estimate해야 한다는 의미인데, MC ES는 policy가 계속 변화함) policy와 value function이 optimal 일 때 비로소 stability를 얻을 수 있다. **MC ES가 정말 optimal로 converge 하는가는 수학적으로 증명되지 않았다..** (for a partial solution, see Tsitsiklis, 2002)


## 5.4 Monte Carlo Contol without Exploring Starts

----

지금까진 계속 exploring starts를 가정해왔다. 어떻게 하면 이 가정을 없앨 수 있을까? 이걸 알아보기 전에 먼제 on-policy와 off-policy가 무엇인지 알아보자.

- on-policy: sample episode 만들때 decision 하는데 사용한 policy를 evaluate고 improve 함
- off-policy: sample episode 만들때 decision 하는데 사용한 policy와 evaluate하고 improve하는 policy가 다름

일례로 MC ES는 on-policy method다. 지금 section 에선 MC ES와 같은 on-policy MC method들이 exploring starts 없이 어떻게 디자인될 수 있을지 알아본다. off-policy method 는 다음 section 에서 다룬다.

on-policy control methods에서 policy는 *soft* ($\pi(a \vert s) \gt 0\text{ for all }s \in S\text{ and all }a \in A(s)$)하다. on-policy는 soft하면서도 점진적으로 deterministic optimal policy로 다가간다. Ch2에서 우리는 이러한 on-policy들을 많이 다뤘었다. 우리는 $\epsilon -greedy$ policies 를 이용할것이다. 이 $\epsilon -greedy$ policies 는 $\epsilon -soft$ policy이다. 여기서 $\epsilon -soft$ 는 $\pi(a \vert s) \ge \frac{\epsilon}{A(s)}$ for all states and actions, for some $\epsilon \gt 0$ 임을 의미한다.

on-policy MC control의 전반적인 idea는 여전히 GPI 이다. MC ES 처럼 우리는 또 first-visit MC Methods를 이용하여 현재 policy의 action-value function을 estimate할 것이다. 그런데 지금은 policy를 전처럼 그냥 greedy하게 improve하지 못한다. 그냥 greedy 하게 improve 했다가는 영영 알 수 없는 state-action pair 들이 생기기 때문에 다른 방법을 강구해야만 한다. 운좋게도 GPI는 policy improve 가 꼭 Greedy여야만 하는게 아니라 다른 방법(greedy로 향하는)을 써도 된다. 따라서 우리는 $\epsilon -greedy$를 이용하여 improve 한다. for any $\epsilon -soft$ policy $\pi$, $q_\pi$에 대한 $\epsilon -greedy$ policy는 $\pi$ 보다 나쁘지 않은게 보장된다. 알고리즘은 아래와 같다

### On-policy Monte Carlo control, for $\epsilon - soft$ policies, estimates $\pi \approx \pi_\ast$

initialize:<br>
&emsp;$\pi(s) \leftarrow$ an arbitrary $\epsilon -soft$ policy<br>
&emsp;$Q(s, a) \in \mathbb{R}$, for all $s \in S, a \in A(s)$<br>
&emsp;$Returns(s, a) \leftarrow$ Empty list, for all $s \in S, a \in A(s)$<br>

Loop forever(for each episode)<br>
&emsp;Generate an episode from $S_0, A_0$, following $\pi: S_0, A_0, R_1, \cdots S_{T-1}, A_{T-1}, R_{T}$<br>
&emsp;G $\leftarrow$ 0<br>
&emsp;Loop for each step of episode, t=T-1, T-2, ..., 0:<br>
&emsp;&emsp;G $\leftarrow \gamma$G + $R_{t+1}$<br>
&emsp;&emsp;Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1, ..., S_{t-1}, A_{t-1}$:<br> 
&emsp;&emsp;&emsp;Append G to $Returns(S_t, A_t)$<br>
&emsp;&emsp;&emsp;$Q(S_t, A_t) \leftarrow average(Returns(S_t, A_t))$<br>
&emsp;&emsp;&emsp;$A^\ast \leftarrow arg\max_a Q(S_t, a)$<br>
&emsp;&emsp;&emsp;for all $a \in A(S_t)$<br>
&emsp;&emsp;&emsp;&emsp;$\pi(a \vert S_t) \leftarrow \begin{cases} 1-\epsilon + \frac{\epsilon}{\vert A(S_t) \vert} & if\ a\ = A^\ast\\ \frac{\epsilon}{\vert A(S_t) \vert}, & if\ a\ \neq A^\ast \end{cases}$<br>

let $\pi'$ be the $\epsilon -greedy$ policy. the conditions of the policy improvement theorem apply because for any $s \in S$:

$$q_\pi (s, \pi'(s)) \ge v_\pi (s)$$

(증명은 책에 있다. 생략.) 즉 $\pi' \ge \pi$ 므로 policy improvement에 써먹을 수 있다. 

## 5.5 Off-policy Prediction via Importance Sampling

----

