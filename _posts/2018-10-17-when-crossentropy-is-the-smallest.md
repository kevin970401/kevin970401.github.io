---
layout: post
title: "when cross entropy loss is the smallest"
categories: MATH
author: lee gunjun
---

# When Cross entropy is the smallest?

Cross Entropy 는 machine learning 또는 information theory 에서 중요한 내용이다.

true label p와 prediction q가 있을 때 ($\sum_i p_i = 1$, $\sum_i q_i = 1$) Cross Entropy 는

$$ H(p, q) = - \sum_{i} p_i * log(q_i) $$

로 주어진다.

여기서 우리는 true label이 0 또는 1로 주어질 때 Cross Entropy 가 최소가 되는 때는 prediction이 label과 같을 때인 것은 쉽게 알 수 있다.

예를 들어 object 가 class A에 속한다 하면, label A 이 1, class B의 label이 0 이고 이 때 cross entropy가 최소가 되는 때는 prediction이 label과 같이 class A에 대해 1, class B에 대해 0일 때이고 이 때의 cross entropy 는 0 이 얻어진다.

하지만 머신러닝을 하다보면 true label 이 0 또는 1로만 주어지는 경우가 아닌, 0과 1 사이의 실수로 주어지는 soft label 을 종종 보게된다. 이 때 우리는 자연스럽게 다음과 같은 궁금증을 가진다

*true label이 soft label 일 때도 prediction이 true label과 같을 때가 Cross Entropy 가 최소일까? 그리고 그 값은 0 일까?*

우선 결론부터 말하자면, **true label 이 soft label 일 때도 prediction이 true label과 같을 때가 Cross Entropy 가 최소가 되는 지점이다. 하지만 그 때의 Cross entropy 값은 true label의 entropy 와 같다.**

그럼 증명을 해보자.

## Proof

----

### 방법 1

----

만약 

$$H(p, q) \ge H(p, p)$$

이면 q=p 일 때 cross entropy 가 최소이며, 그 외의 상황에서 항상 $H(p, q) \ge H(p, p)$ 임을 알 수 있다.

참고로 H(p, p)는 p의 entropy H(p) 와 같다.

오른쪽 항을 왼쪽으로 이항하자

$$\begin{matrix}
H(p, q) - H(p, p) & = &  - \sum_{i} p_i * log(q_i) + \sum_{i} p_i * log(p_i)\\
& = & \sum_{i} p_i * -log(\frac{q_i}{p_i})\\
& \ge & -log(\sum_{i} \frac{q_i}{p_i} * p_i) \space\space\space \cdots\ by\ jensen\ inequality \\
& = & -log(\sum_{i} q_i)\\
& = & -log(1)\\
& = & 0\\
\end{matrix}$$

$$\therefore H(p, q) \ge H(p, p)$$

또한 젠센 부등식의 등호 성립 조건에 의해 H(p, q)가 최소인 때는 q=p 일 때임을 알 수 있다. 그리고 이때 cross entropy 의 값은 p 의 entropy 와 같고 H(p, p) 가 된다.

### 방법 2 (엄밀하지 않음. 의미 파악 좋음)

----

Cross entropy는 entropy 와 KL-Divergence 로 나눌 수 있다. 즉

$$H(p, q) = H(p, p) + D_{KL} (p || q)$$

이다.

이때 KL-Divergence 의 특성을 통해 $D_{KL} (p \vert \vert q)$ 는 p=q일 때 최소이고 그때의 값은 0 임을 알 수 있다. (이것을 증명하고 싶으면 위와 마찬가지로 젠센 부등식을 사용하면 된다.)

H(p, p)는 constant 므로 q=p 일 때 H(p, q) 은 최소고 이때 cross entropy의 값은 p의 entropy 인 H(p, p) 이다.

## 마무리

----

사실 cross entropy 의 의미를 떠올린다면 H(p, q) 가 최소가 되는 때가 언제인지는 매우 쉽게 알 수 있는 내용이다. 그럼에도 가끔씩 그런 걸 까먹는 나같은 사람들을 위해 포스팅 해봤다.