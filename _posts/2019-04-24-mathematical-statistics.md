---
layout: post
title: "수리통계학 (김우철)"
categories: ETC
author: lee gunjun
---

# 수리통계학 (김우철)

---

## Summary

### ch3 여러가지 확률 분포

**초기하분포**
> 크기가 N인(성공 D, 실패 N-D) 유한모집단에서 비복원 단순랜덤추출(simple random sampling) 으로 n개의 sample을 취할 때 성공한 횟수 X
>
> $f_X(x) = \binom{D}{x}\binom{N-D}{n-x}/\binom{N}{n}$

**이항분포**
> 초기하분포에서 비복원추출을 복원추출로
>
> $X \sim B(n, p)$
>
> $\leftrightarrow f_X(x) = \binom{n}{x}p^x(1-p)^{n-x}$

**베르누이분포**
> 이항분포에서 n=1
>
> $X \sim Bernoulli(p)$
>
> $\leftrightarrow P(X=1) = p, P(x=0) = 1-p, 0 \le p \le 1$

**다항분포**
> 이항분포 확장. 이젠 결과가 0, 1 만이 아니라 2, 3, ... 도 가능
>
> 모집단에서 각 유형의 비율이 $p_1, p_2, \cdots, p_k$인 경우, 복원추출한 n개의 랜덤표본에 있는 각 유형의 개수를 $X_1, X_2, \cdots, X_k$ 라 하면, 결합밀도함수는
>
> $f(x_1, x_2, \cdots, x_k) = \binom{n}{x_1, x_2, \cdots, x_k}p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k},\ where\ x_1+x_2+\cdots+x_k=n$
>
> $X = (X_1, X_2, \cdots, X_k)^t \sim Multi(n, (p_1, p_2, \cdots, p_k))$
>
> 다항분포 역시 $Z_i \sim Multi(1, (p_1, p_2, \cdots, p_k))$ 인 $Z_i$ n개의 합으로 나눌 수 있다.

**기하분포**
> 서로 독립이고 성공률이 p인 베르누이 시행 $X_1, \cdots, X_n, \cdots$ 을 관측할 때, 첫번째 성공까지의 시행횟수를 $W_1$ 라 하면 $W_1$ 는 기하분포를 따르고
>
> $P(W_1 = x) = (1-p)^{x-1}p$
>
> $\leftrightarrow W_1 \sim Geo(p)$

**음이항분포**
> 서로 독립이고 성공률이 p인 베르누이 시행 $X_1, \cdots, X_n, \cdots$ 을 관측할 때, r번째 성공까지의 시행횟수를 $W_r$ 라 하면 $W_r$ 는 음이항분포를 따르고
>
> $P(W_r = x) = \binom{x-1}{r-1}p^{r}(1-p)^{x-r}$
>
> $\leftrightarrow W_r \sim Negbin(r, p)$

**푸아송분포**
> 이항분포에서 $n \rightarrow \infty,\ np \rightarrow \lambda$
>
> $\underset{n \rightarrow \infty,\ np \rightarrow \lambda}{lim} \binom{n}{x}p^x (1-p)^{1-x} = \frac{e^{-\lambda}\lambda^x}{x!},\ where\ \lambda \gt 0$

**푸아송과정**
> 4가지 조건(stationary, independent increment, proportionality, rareness)을  만족하는 과정을 푸아송 과정이라 함.
>
> 발생률(occurrence rate) $\lambda$ 인 푸아송 과정 {$N_{t} : t \gt 0$} 에서 시각 t 까지 발생횟수 $N_t$ 의 분포는 평균이 $\lambda$ 인 푸아송분포이다.
>
> $N_t \sim Poisson(\lambda t)$

**지수분포**
> 푸아송과정에서 첫 번째 현상이 발생할 때까지의 시간 $W_1$ 의 분포를 지수분포라 함.
>
> $f(x) = \lambda e^{-\lambda x}I_{(x \ge 0)}$
>
> $W_1 \sim Exp(1/\lambda)$

**감마분포**
> 푸아송과정에서 r번째 현상이 발생할 때까지의 시간 $W_r$ 의 분포를 감마분포라 함.
>
> $f(x) = \frac{1}{\Gamma(\alpha)\beta^{\alpha}}x^{\alpha-1}e^{-x/\beta} I_{(x \gt 0)}$
>
> $W_r \sim Gamma(r, 1/ \lambda)$
>
> 지수분포 r개의 합으로 표현가능. $X \equiv Z_1+\cdots+Z_r,\ Z_i \overset{iid}{\sim} Exp(1/ \lambda)$

**정규분포**
> $Z \sim N(\mu, \sigma^2)$
>
> $\leftrightarrow p(z)=\frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

#### 분포들의 관계

* 이항분포는 베르누이분포 n개로 나눌수 있음. ($X \equiv Z_1 + \cdots + Z_n$)
* 음이항분포는 r개의 기하분포로 나눌수 있음.
* 정규분포의 제곱은 감마분포 Gamma(1/2, 2) 임.

### ch4 표본분포

모수: $\theta$ (모집단 분포를 결정하는 특성치)

모수공간: $\Omega$ (모수들의 집합)

모집단 분포: $f(x; \theta)$ (pdf)

**랜덤표본과 통계량**
> 모집단 분포가 pdf $f(x; \theta), \theta \in \Omega$ 인 모집단에서의 random sample $X_1, X_2, \cdots, X_n$ 은
>
> $X_1, X_2, \cdots, X_n \overset{iid}{\sim} f(x; \theta),\ \theta \in \Omega$ 이다.
>
> 그리고 통계량 $u(X_1, X_2, \cdots, X_n)$ 은 random sample의 함수

**치환 pdf**
> 1. $P(X \in \mathcal{X}) = 1$
> 2. $u = (u_1, \cdots, u_k)^t$ : $\mathcal{X} \rightarrow \mathcal{Y}$ 는 정의역이 $\mathcal{X}$ 이고 치역이 $\mathcal{Y}$인 m대1 함수
> 3. 정의역을 disjointed open set들의 합으로 표현 가능하고 그 각 set에서 일대일 함수이다.
> 이 때 $Y = u(X) 즉 Y = (Y_1, \cdots, Y_k)^t = (u_1(X), \cdots, u_k(X)) 의 pdf는$
>
> $pdf_Y(y) = \underset{x: u(x) = y}{\sum} pdf_X(x) \left\vert det(\frac{\partial u(x)}{\partial x}) \right\vert^{-1}, y \in \mathcal{Y}$

#### 여러 분포

**균등분포**
> $X \sim U(a, b)$
>
> $\leftrightarrow pdf_X(x) = \frac{1}{b-a}I_{(a, b)}(x)$

**베타분포**
> $X \sim Beta(\alpha_1, \alpha_2) \leftrightarrow X \equiv \frac{Z_1}{Z_1 + Z_2}$
>
> $Z_i \sim Gamma(\alpha_i, \beta),\ i=1,2$
>
> $pdf_X(x) = \frac{\Gamma(\alpha_1 + \alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)}x^{\alpha_1-1}(1-x)^{\alpha_2-1}I_{(0, 1)}(x)$

**디리클레분포**
> 베타분포를 다차원으로 일반화
>
> $Y = (Y_1, \cdots, Y_k)^t \sim Dirichlet(\alpha_1, \cdots, \alpha_k, \alpha_{k+1})$
>
> $\leftrightarrow Y \equiv (\frac{X_1}{X_1+\cdots+X_{k+1}}, \cdots, \frac{X_k}{X_1+\cdots+X_{k+1}}),\ where\ X_i \sim Gamma(\alpha_i, \beta), 서로\ 독립$

-----

여기서부터는 모집단의 분포가 정규분포임을 가정하는 경우에 사용되는 통계량인 표본평균과 표본분산의 분포를 나타내는 대표적인 표본분포들에 대해 알아본다.

**카이제곱분포**
> $Y \equiv X_1^2 + \cdots + X_r^2, X_i \overset{iid}{\sim} N(0,1)$
>
> $\leftrightarrow Y \sim \chi^2(r),\ where\ (r>0)$
>
> $Y \sim Gamma(r/2, 2)$ ($\because X_i \sim Gamma(1/2, 2)$)
>
> 감마함수임.

**t 분포**
> $X \equiv \frac{Z}{\sqrt{V/r}},\ where\ Z \sim N(0, 1),\ V \sim \chi^2(r),\ Z와\ V는\ 독립$
>
> $\leftrightarrow X \sim t(r),\ where\ (r>0)$
>
> 특징: $pdf_X(x) = pdf_X(-x)$
>
> $P(X \gt t_\alpha (r)) = \alpha$ 를 만족하는 $t_\alpha (r)$ 를 $\alpha 분위수라 함$
>
> 많이 쓰이는 부등식: $P(\left\vert X \right\vert \gt t_{\alpha/2} (r)) = \alpha$

**F 분포**
> $X \sim F(r_1, r_2) (r_i \gt 0, i=1, 2)$
>
> $\leftrightarrow X \equiv \frac{V_1/r_1}{V_2/r_2},\  V_i \sim \chi^2(r_i),\ i=1,2$
>
> 특징 1: $F_{1-\alpha}(r_1, r_2) = 1/F_\alpha (r_2, r_1)$
>
> 특징 2: $t_{\alpha / 2}^2 (r) = F_\alpha (1, r)$

**다변량 정규분포**
> $Z \sim N_n(0, I) \leftrightarrow Z=(Z_1, \cdots, Z_n)^t,\ Z_i \overset{iid}{\sim} N(0, 1), i=1, \cdots, n$
>
> $X \sim N_n(\mu, \Sigma)$ 
>
> 1. $\leftrightarrow X \equiv AZ + \mu, Z \sim N(0, 1), AA^t = \Sigma$
> 2. $\leftrightarrow X \equiv \Sigma^{1/2}Z + \mu, \Sigma^{1/2}\Sigma^{1/2}=\Sigma, \Sigma^{1/2}=(\Sigma^{1/2})^t$
> 3. $pdf_X(x) = \frac{1}{\sqrt{det(2\pi\Sigma)}} e^{-\frac{1}{2}(x-\mu)^t \Sigma^{-1} (x-\mu)}$
>
> 특징 1: $AX+b \sim N(A\mu+b, A\Sigma A^t)$
>
> 특징 2: $\dbinom{X_1}{X_2} \sim N \left ( \dbinom{\mu_1}{\mu_2}, \begin{pmatrix} \Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22} \end{pmatrix} \right )$ 일 때, $Cov(X_1, X_2) = \Sigma_{12} = 0$ 이면 $X_1$와 $X_2$ 는 서로 독립
>
> 특징 3: $\dbinom{X_1}{X_2} \sim N \left ( \dbinom{\mu_1}{\mu_2}, \begin{pmatrix} \Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22} \end{pmatrix} \right )$ 면 $X_1 \sim N(\mu_1, \Sigma_{11})$
>
> 특징 4: $X \sim N_k(\mu, \Sigma)$ 이고 $\Sigma$가 정칙행렬이면 
>
> $$(X-\mu)^t \Sigma (X-\mu) \sim \chi^2(k)$$

### ch5 표본분포의 근사

**중심극한정리**
> 확률변수 $X_1, X_2, \cdots, X_n$ 이 서로 독립이고 동일한 분포를 따르며 $Var(X_1)$ 이 양의 실수일 때
>
> $$E(X_1) = \mu, Var(X_1) = \sigma^2 < \infty$$

#### 분포 간의 관계

* 카이제곱 분포는 감마분포
* 정규 분포의 제곱은 카이제곱분포 ($Y = X^2 \sim \chi(1) = Gamma(1/2, 2)$)
* 카이제곱 분포 나누기 카이제곱 분포는 F 분포

#### 정규모집단에서 모평균/모분산 의 추론
* 모평균 추론
    * $\frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t(n-1)$ 이용 (정리 4.2.3)
    * $P\{ \bar{X} - t_{\alpha/2}(x-1)S / \sqrt{n} \le \mu \le \bar{X} + t_{\alpha/2}(x-1)S / \sqrt{n} \} = 1-\alpha$
* 모분산 추론
    * $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$ 이용 (정리 4.2.2)
    * $P\{\frac{n-1}{\chi_{\alpha/2}^2 (n-1)} S^2 \le () \le \frac{n-1}{\chi_{1-\alpha/2}^2 (n-1)} S^2 \} = 1- \alpha$

#### 두 정규모집단에서 모분산의 비교
$X_{1i} \sim N(\mu_1, \sigma_1^2), where\ i=1, 2, \cdots, n_1\ and\ X_{2i} \sim N(\mu_2, \sigma_2^2),\ where\ i=1, 2, \cdots, n_2$ 일 때

$$\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} \sim F(n_1-1, n_2-1)$$

$$P\{\frac{S_1^2}{S_2^2} \frac{1}{F_{\alpha/2} (n_1-1, n_2-1)} \le \frac{\sigma_1^2}{\sigma_2^2} \le \frac{S_1^2}{S_2^2} F_{\alpha/2} (n_2-1, n_1-1) \} = 1 - \alpha$$

#### 순서통계량
모집단 분포가 연속형이고 그 확률밀도함수가 $f(x)$ 일 떄, 핸덤 표본 $X_1, X_2, \cdots, X_n$ 을 크기 순서로 늘어놓은 순서통계량을 $X_{(1)} \lt X_{(2)} \lt \cdots \lt X_{(n)}$ 이라고 하면 $Y=(X_{(1)}, \cdots, X_{(n)})$ 결합확률밀도함수는 다음과 같다.

$$pdf_Y(y_1, y_2, \cdots, y_n) = n! f(y_1) \cdots f(y_n) I_{(y_1 \lt \cdots \lt y_n)}$$

#### 선형회귀모형
$Y_i = x_{i0}\beta_0+\cdots+x_{ip}\beta_p+e_i, e_i \overset{iid}{\sim}N(0, \sigma^2I), i=0, \cdots, n$

표본회귀계수: $\hat{\beta} = (X^t X)^{-1}X^t Y$

평균오차제곱합: $\hat{\sigma^2} = (Y-X \hat{\beta})^t (Y-X \hat{\beta}) / (n-p-1)$

$\hat{\beta} \sim N_{p+1}(\beta, \sigma^2(X^t X)^{-1})$

$(n-p-1)\hat{\sigma^2}/\sigma^2 \sim \chi^2(n-p-1)$


****
## 좋은/어려운 문제

1. Chapter 1
    - 연습문제 1-20
