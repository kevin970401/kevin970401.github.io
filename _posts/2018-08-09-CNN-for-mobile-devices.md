---
layout: post
title: "CNN for mobile devices"
categories: CNN
author: lee gunjun
---

# Model architecture

[SqueezeNet, X](https://arxiv.org/pdf/1602.07360)

3가지 strategies
1. 3x3 filter 를 1x1 filter 로 대체 (param 줄임)
2. 3x3 filter 의 input 의 channel 을 줄임 (param 줄임)
3. downsample late -> large activation map (accuracy 높임)

3x3 filter 는 spatial 정보를 씀.

fire module 을 소개. squeeze(1x1 filters) 와 expand(1x1, 3x3 mixture) 로 구성. 
squeeze 는 channel 줄이고 expand 에서 다시 늘림. 

뭐 3x3 1x1 비율을 pct 라는 변수로 조절하고, 각 fire module 의 output channel을 변수로 조절하는 등의 시도가 있는데, 중요한 거 아니니 생략.

1. vanila
1. bypass(maxpool 로 downsample 하는 구간 빼고 bypass)
1. complex(maxpool 로 downsample 하는 구간 빼고 1x1 conv 한번 태워서 bypass)

해봤는데 bypass 가 제일 좋았음

[MobileNetV1, X](https://arxiv.org/pdf/1704.04861.pdf)

depthwise separable convolution: depthwise convolution + 1x1 pointwise convolution  
depthwise separable convolution has the effect of drastically reducing computation and model size.

original conv 를 spatial, cross-channel 로 factorize 한 것 (xception 의 아이디어와 동일)

fearture map: $(D_F, D_F, M)$->$(D_F, D_F, N)$  
filter: $(D_K, D_K, M, N)$  
이면

original:  
computation: $D_F*D_F*D_K*D_K*M*N$  
`#` of params $D_K*D_K*M*N$

depthwise separable convolution:  
computation: $D_F*D_F*D_K*D_K*M + D_F*D_F*M*N$  
`#` of params: $D_K*D_K*M+M*N$

대부분의 param과 computation 은 pointwise conv 에 몰려있음.(param은 75%, computation은 95%)

depthwise conv, pointwise conv 각각 bn, relu 적용

first layer 는 conventional convolution layer 씀. downsample 여기서 한번 해줌.

fc 전에 global average pooling 함.


two hyperparams trade off latency and accuracy: width multiplier, resolution multiplier

width multiplier($\alpha$): the number of input channels $M$ becomes $\alpha M$ and the number of output channels $N$ becomes $\alpha N$. reducing both computation and # of params by roughly $\alpha^2$

resolution multiplier($\rho$): input image size 를 $\rho$ 만큼 줄여서 넣음. 당연히 중간 layer activate map 도 $\rho$ 만큼 줄어듦.

[ShuffleNetV1, CVPR 2018](https://arxiv.org/pdf/1707.01083.pdf)

resnet 의 변형.

두개의 idea.

1. 1x1 convolution 의 computation and param 을 줄이기 위해 pointwise group convolution 을 제시. 
1. group conv 의 side effect(group conv 하고 다시 group conv 하면 한번 잘린 group 은 계속 유지 됨. 두 group conv 가 같은 g가 아니어도 충분히 문제가 됨. ~~글로 적기 어렵다~~) 를 막기 위해 channel shuffle operation 도입. 이를 통해 channel 간의 information flow 가 원활히 되도록 함.

pointwise group convolution: 1x1 conv 을 group conv 으로 함. computation 1/g 으로 줆

channel shuffle operation: group conv 결과로 g*n 개 channel 이 output 으로 나오면 이를 (g, n) 으로 만들고 (n, g) 으로 transpose 한 다음 flatten 해서 (n*g) 로 만들면 끝.

shuffle unit 제안함. mobilenet 과는 달리 bypass 씀. feature map 1/2 로 줄일땐 channel 2배로 하기 위해 skip connection 한걸 element-wise add 가 아니라 concat 함.

끝

[CondenseNet, CVPR 2018](https://arxiv.org/pdf/1711.09224.pdf)

먼저 densenet 을 먼저 알아야함. densenet 은 denseblock 을 여러개 이어서 만듦. 한 denseblock 내에서는 모든 layer 의 feature map size 가 같음. 한 block 내의 각 layer들 결과물은 뒤의 모든 layer 에 concat 됨. 각 layer 의 output channel 개수를 growth rate (k)라 함. 맨 마지막 layer는 $k_0 + k * (l-1)$ 개의 input channel 가짐. 한 layer 지날때 마다 input channel 개수가 k씩 증가하므로 growth rate 라 부름. k 가 작아도 학습 잘 됨.

feature 가 계속 쌓이니까 이를 해결하기 위해 bottleneck layer 도입. concat 한다음 1x1 conv 로 channel 줄이고 3x3 conv 에 넣음. 더불어 transition layer 를 도입하는데, bottleneck layer는 block내의 channel 을 줄이는 역할이었다면 transition layer 는 block output의 chanenl 줄임.

이제 condensenet 보자.

1x1의 group conv 는 drastic reductions in accuracy 를 가져옴.(shufflenet 과 같은 구조에선 잘 모르겠지만 densenet의 1x1 conv는 concat 된 output에 적용하는 거라 심하게 나빠짐)

그렇다고 group conv 를 random permutation 을 하고 쓴다하더라도, 줄어든 computation, param 만큼 layer 수 줄인 그냥 densenet 이 더 잘 됨.

어떻게 group conv 를 accuracy 손해 없이 적용할 수 있을까 생각함. 그래서 input feature grouping 을 automatically learn during training 하는 방식을 고안하게 됨.

learning group conv 는 multi-stage 임. 첫 stage 는 condensing stage: 학습 하면서 unimportant filter 는 없앰. 다음 stage 는 optimization stage: 다른 group 간의 filter 는 input feature 가 겹치지 않게끔 filter 를 pruning 함. 예를 들어 output feature group 0 에 있는 output feature 를 만드는데 사용하는 filter 가 0번 input feature 를 사용하는데, output feature group 1 에 있는 filter가 또 0번 input feature 를 사용하면 안 됨. 설명하기 어려우니 논문 Fig 3 보자. 이렇게 하면 pruning 결과는 평범한 group conv 가 됨. group conv 는 여러 딥러닝 library 에 효과적으로 구현되어 있으므로 굳이 group conv 를 만들어 주는 것임. 아무렇게나 filter 단위로 쪼개져 있어도 효과적으로 operation 되면 굳이 group conv 안쓰겠지

위 알고리즘을 자세히 살펴보자. 

1 filter groups

통상적 conv filter 는 $C_{in}*C_{out}*H*W$ 형태다. 근데 우리가 지금 다루는 conv 는 1x1이므로 그냥 $(C_{out}, C_{in})$ 이라 봐도 됨. 먼저 training 하기 전에 filter 를 g개 group 으로 나눔. $F^1, \cdots, F^g$ 각 $F$ 는 $(\frac{C_{out}}{g}, C_{in})$ 임. 

2 Condensation Criterion

training 하면서 덜 중요한 input feature 를 각 group 마다 제거함. 제거 알고리즘: 각 group 의 column 별로 L1-norm 을 구하고 norm 이 가장 작은 column 을 제거. 즉 group 내에서 가장 안쓰이는 input feature channel 을 없앤다는 의미다. 이렇게 pruning 을 한 group 내의 output channel 들은 input channel 을 공유할 것이다. 그리고 한 input channel 이 여러 output group 으로 갈 수 있다. 중복 되는 애들은 group conv 를 위해 replicate 함.

2-1 Group Lasso

loss에 L1 regularization 을 도입한다. 다만 `2.`에서 하는 pruning 의 accuracy 부정적 영향을 없애기 위해 각 column 별로 l1-norm 을 계산하여 적용한다.

$$group lasso = \sum_{g=1}^G \sum_{j=1}^{C_in} \sqrt{\sum_{i=1}^{\frac{C_{out}}{g}}{F_{i,j}^g}^2}$$

3 Condensation Factor: $C$

사실 여기선 g 가 딱 $C_{out}, C_{in}$ 의 약수일 필요 없음. each group 이 $C=\lfloor \frac{C_{in}}{C} \rfloor$ 개 의 input feature 를 선택하게 함.

4  Condensation Procedure

우리 weight pruning 은 conventional pruning 과 다르게 training 후에 하는 게 아니라 training 하면서 함. $C-1$ 개의 condensation stage 동안 $\frac{1}{C}$ 만큼의 input feature 를 pruning 하면 최종적으로 $\frac{1}{C}$ 만큼의 input feature 가 남음.

group convolution 도입을 통해 densenet 에 변화를 두가지 더 줌.

1.  growth rate 를 exponentially increase 함. 가까운 layer에 더 큰 concat 가중치를 주기위해 $k=2^{m-1}k_0$ 로 exponential 하게 줌.
2.  fully dense connectivity: dense block 없이 한번에 함. feature size 가 다르면 downsample 해서 맞춤.

결과론적으로 group conv 에 의한 err 증가를 낮출 수 있었고, 최종적으로 model param 개수를 동일하게 하면 전보다 더 err를 낮출 수 있음

[MobileNetV2, CVPR 2018](https://arxiv.org/pdf/1801.04381.pdf)

mobilenetv2 을 활용하여 detection 으로는 SSDLite, segmentation 으로는 Mobile DeepLabv3 이 파생되어 나옴.

1. mobilenet v1 에서 나온 depthwise separable convolution 계속 씀.
2. linear bottlenecks

<!-- resnet 와 같은 곳에서 bottleneck 은 dimension 을 1x1 conv 로 줄이고 다시 늘리는 형태였다. 이런 형태에는 문제가 있다.

이를 설명하기 위해선 그 전에 manifold, relu 에 대해 알아봐야한다.

layer의 activation tensor $L_i$ 는 manifold of interest 를 형성한다. 즉 모든 feature map 을 고려한다 해도 결국 중요한 정보는 subspace 에 놓여있다. 이런 논리하면 분명 layer 의 dimension 을 manifold 의 dimension 까지는 낮춰도 성능이 떨어지지 않을것이다.

layer 의 dimension 을 줄이기 위해 mobilenet v1 에서는 width multiplier 를 제시했었다. width multiplier 는 activation space dimension 을 manifold of interst 가 span하여 만들 수 있는 공간까지 낮추도록했다. -->

<!-- 그런데 여기서 문제가 있다! 우리는 ReLU 를 쓴다. deep networks only have the power of a linear classifier on the non-zero volume part of the output domain. 뿐더러 ReLU 는 channel 의 information 을 잃어버리게 한다. 근데 우리가 channel 이 매우 많다면 어느 channel 에서 잃어버린 information 을 다른 channel 에서 복구해줄것이라 기대할 수 있다. 그러나 mobile 에선 channel 수를 줄여야 하기 때문에 이를 다른 방식으로 해결할 필요가 있다.

그래서 linear bottleneck 을 제안한다. 평범한 bottleneck에서 relu 를 제거한 것이다. -->

우리는 parameter, computation 을 줄이기 위해 bottleneck 을 사용하는데, 기존의 bottleneck(relu가 conv 뒤에 있는)은 relu에서 두개의 문제점이 생긴다. 첫번째 단점은 두번째 단점은 . 이를 해결하기 위해서는 많은 channel 을 써야하는데 channel 수를 적게 써야하는 bottleneck 에서는 사용할 수 없는 방법이니 다른 해결방법을 도색해야했다. 그래서 그냥 우리는 relu 를 사용하지 않기로 했고 그것이 linear bottleneck 이다. 

(사실 bottleneck에서 relu 에 대한 문제제기는 이미 Deep Pyramidal Residual Networks(CVPR 2017) 에 나왔었다.)

Inverted residuals

보통의 bottleneck block 은 input 이 들어오면 bottleneck 에서 channel 을 줄이고 conv 하고 expansion 을 하는데 이는 input 과 output channel 이 크고 중간 과정의 channel 수가 적은 구조이다. 그렇기에 여기에 residual 연산을 하게 되면 많은 computation 과 memory 를 사용한다. mobilenet v2 에서는 이를 비효율적인 과정으로 생각하고 효과적으로 바꾸기 위해 inverted residual 을 제안한다. 이는 input 을 먼저 expansion 하고 squeeze 하는 구조로 residual 과정의 computation 과 memory 를 줄여준다. 

여담: mobilenet v2 는 재현이 잘 안된다는 의견이 꽤 있음. imagenet 에 overfitting 된 모델로 추정됨.

[SqueezeNext, CVPR 2019](https://arxiv.org/pdf/1803.10615.pdf)

mobilenet 의 depthwise-separable conv 는 몇몇 mobile 에서 비효율적으로 동작한다.

squeezenet 에서 3가지 발전

1. more aggresive channel reduction by incorporating a two-stage squeeze module
2. use separable 3x3 conv, remove the additional 1x1 branch
3. skip connection like resnet

----

network 를 compression 하는 방법으로는 post-training compression 과 pre-training compression 이 있음. post-training compression 에는 reduced precision, pruning to reduce the number of non-zero weights 과 같은 학습된 network 을 low rank로 만들어주는 작업들이다. 그러나 이들은 앞선 연구자들의 많은 경험을 통해 retraining 혹은 accuracy 의 저하와 같은 부작용을 피할 수 없다는 게 중론이다. 그래서 우리는 pre-training compression 을 다룰 것이다. 애초에 작은 model 을 써서 low rank 을 강제하는 방법이 있는데, 우리가 사용할 방법이다.

filter 의 parameter 를 줄이는 방법을 알아보자.

conv 의 \# of parameters: $C_{in}K^2C_{out}$

1. $K^2$ 을 줄이기 위해 KxK conv 를 Kx1, 1xK conv 로 factorize 한다. 각각 relu와 bn 을 가짐. (inception 에서 쓴 방법)
2. $C_{in}C_{out}$ 을 줄이기 위해 depthwise-separable conv 를 써서 parameter 를 줄이는 방법이 있지만 이는 몇 embedded system 에서 좋은 performance 를 보여주지 않는다. squeezenet의 squeeze layer 를 쓰는 방법이 있다. squeezenext 에서는 이를 2-stage 로 함.

그리고 fc 가 weight 를 굉장히 많이 가지는 layer 인데 이를 해결하기 위해 fc layer 전에 squeeze layer 를 둠.

hardware: nn accelerator 를 말함. simulator 를 만들어서 op 별 profiling 함. 그걸 토대로 model 를 바꿈.

[ShuffleNetV2, ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)

지금까지 metric으로는 computation complexity(=indirect metric) 를 중점으로 했는데, 사실 우리가 정말 원하는 건 속도다(=direct metric). memory access cost(=MAC) 혹은 degree of parallelism 과 같은 것들이 단순 FLOPS metric 에는 고려되지 않음. 뿐더러 platform-dependent 한 부분들도 있음.

shuffle v2 에서는 속도를 위한 몇가지 가이드를 제시함.

G1. Equal channel width

1x1 에서 flops 는 B = $hwc_{in}c_{out}$, MAC은 $hw(c_{in}+c_{out})+c_{in}c_{out}$. 같은 flops 일때 mac이 최소가 되려면 $c_{in}$, $c_{out}$이 같아야 함.

G2. Group Conv increases MAC

FLOPS B = $\frac{hwc_{in}c_{out}}{g}$, MAC = $hw(c_{in}+c_{out})+\frac{c_{in}c_{out}}{g} = hwc_{in}+\frac{Bg}{c_{in}}+\frac{B}{hw}$

G3. Network fragmentation reduces degree of parallelism

Network fragmentation은 쉽게 factorizing 말함.

G4. Element-wise operations are non-negligible

FLOPS 는 적지만 MAC이 큼. depthwise conv 도 FLOPS 에 비해 MAC이 큼.

끝

가이드 안지키는 예시로 ShuffleNet v1 는 G1, G2 에 위배, Mobilenet v2는 G1, G4 위배, auto-generated structure 들은 보통 G3 위배

이 가이드라인을 잘 지키는 block 을 만듦.

input 의 channel의 반을 나눔. G3에 위반하지 않는이유는 반쪽 branch 에는 아무 연산도 하지 않아서 괜찮음. 다른 반쪽 branch 에서는 channel 이 변하지 않아서 G1 만족. 1x1 conv 는 group conv 을 사용하지 않아서 G2 만족. skip connection 에서 add 대신 concat 써서 G4 만족.

반쪽은 아무 연산 안하고 그냥 connect 되는데 이게 densenet과 비슷한 효과를 냄.

[MnasNet, CVPR 2019](https://arxiv.org/pdf/1807.11626.pdf)

flops 나 mac 대신 latency를 reward 에 사용.

reward:  
maximize $ACC(m)$ subject to $LAT(m) \le T$ 대신

maximize $ACC(m) * \left[ \frac{LAT(m)}{T} \right]^w$ where w is the weight factor defined as $w = \begin{cases} \alpha, & \text{if } LAT(m) \le T \\ \beta, & otherwise \end{cases}$ w는 음수. 을 사용

search space: 제한을 많이 줌.

결과: mobilenet v2 와 비교했을 때 같은 acc에서 두배이상 빠름.

<!-- How to Estimate the Energy Consumption of DNNs -->

[MobileNet V3, X](https://arxiv.org/pdf/1905.02244)