---
layout: post
title: "Domain Adaptation(정리 안 됨)"
categories: DNN
author: lee gunjun
---

# Background
----

## overfitting
$hypothesis h \in \mathcal{H}$에 대해 $R_{train}(h) < R_{train}(h')\ \&\ R_{test}(h) > R_{test}(h')$ 을 만족하는 어떤 $h' \in \mathcal{H}$ 가 존재하면 $h$는 overfitting 되었다고 한다.

## Occam's razor
모델의 complexity가 커지면 training error 는 줄어들지만 true error는 증가한다

## PAC (Probably Approximately Correct)
Probably ... model Approximately Corrects

- num of Training data: m
- gap $\epsilon$: $R_{true}(h) \le R_{train}(h) + \epsilon$
- complexity of Hypothesis set $\mathcal{H}$: $\left\vert \mathcal{H} \right\vert$

### PAC Bound
 
$$Pr[R_{true}(h) - R_{train}(h) \gt \epsilon] \le \left\vert \mathcal{H} \right\vert e^{-2m\epsilon^2}$$

일반화 잘 하는 법
- training data sample 수 (=m) 많이
- model complexity (=$\vert \mathcal{H} \vert$) 작게 

## PAC with Infinite Hypothesis Space

$\mathcal{H}$ 가 infinite set 이면 $|\mathcal{H}|$ 가 infinite 가 되어버리므로 PAC bound 가 쓸모가 없어짐.

그래서 나온게 VC(Vapnik-Chervonenkis) dimension 임. $VC(\mathcal{H})$ 는 Instance space $\mathcal{X}$ 와 $\mathcal{H}$ 에 대해 정의됨. 의미론적으론 classification task 에서 $\mathcal{H}$ capacity 를 의미함.

$VC(\mathcal{H})$ 는 $\mathcal{X}$ 에 대해 $\mathcal{H}$ 로 shatter 할 수 있는 $\mathcal{A} \in \mathcal{X}$ 의 가장 큰 크기.

### Example

#### Example - 1

$\mathcal{X}$ = 1-dimension 이고, $\mathcal{H}$ 가 linear model 이면

* num(A)=2 인 $A \in \mathcal{X}$ 에 대해
    * ----O---X---- 구별가능.
* num(A)=3 인 $A \in \mathcal{X}$ 에 대해
    * ---O--O--X--- 구별 가능
    * ---O--X--X--- 구별 가능
    * ---O--X--O--- 구별 불가능..

따라서 $VC(\mathcal{H}) = 2$

#### Example - 2
$\mathcal{X}$ = n-dimension 이고, $\mathcal{H}$ 가 linear model 이면

$VC(\mathcal{H}) = n+1$

증명 어려우니 생략

#### Example - 3
k-Nearest Neighbor 에서 k = 1 일 때

$VC(\mathcal{H}) = \infin$

증명 당연하므로 생략

## $VC$ 와 $\vert H \vert$ 의 relationship

$VC(\mathcal{H}) = k$ 라고 하면 $\mathcal{H}$ 을 통해 k개 instances 를 shatter 할 수 있다는 것. 얘네들을 labeling 하는 가짓수는 총 $2^k$ 므로 $\vert \mathcal{H} \vert \ge 2^k$ 따라서 

$$VC(\mathcal{H}) = k \le log_2(\vert \mathcal{H} \vert)$$

# DANN
----

$$\mathcal{H}-divergence$$
$$\begin{matrix}\\
d_\mathcal{H}(\mathcal{D_S^X}, \mathcal{D_T^X}) & = & 2 \sup_{\eta \in \mathcal{H}} \left\vert \Pr_{x \sim D_S^X}[\eta (x) = 1] - \Pr_{x \sim D_T^X}[\eta (x) = 1] \right\vert \\
& = & 2 \sup_{\eta \in \mathcal{H}} \left\vert \Pr_{x \sim D_S^X}[\eta (x) = 0] - \Pr_{x \sim D_T^X}[\eta (x) = 0] \right\vert \\
\end{matrix}$$

# Reference

- http://sanghyukchun.github.io/66/
- http://www.cs.cmu.edu/~guestrin/Class/10701-S05/slides/pac-vc.pdf