---
layout: post
title: "Video Object Segmentation (정리 안 됨)"
categories: CNN
author: lee gunjun
---

# Video Object Segmentation

----

## 챌린지
----

* DAVIS
    * semi-supervised
        * first frame의 gt mask is given

## 기법
----

*VOS에서 구현 잘 된 코드 진짜 한번을 못봄..*

### [FEELVOS](https://arxiv.org/abs/1902.09513)

Fast End-to-End Embedding Learning for Video Object Segmention

DAVIS 에서 fine-tuning on first frame 없는 기법들 중에선 SOTA.

semi-supervised VOS는 보통 first frame gt mask 에 fine-tuning 하는 ~~꼼수~~ 기법이 많이 쓰이고 더불어 extensive engineering 을 걸쳐 나온 complex 한 model 을 이용한다. 그로 인해 성능은 좋지만 runtime 이 매우 길어 practical 하게 쓰는 데에는 문제가 있었다. *(2018 DAVIS 에서 1위를 차지한 PReMVOS 는 한 frame 당 38초나 걸린다.)*

FEELVOS 는 practical usability 를 고려함.

#### Practical Usability 의 요소

* Simple
    * single NN
* Fast
    * not rely on first-frame fine-tuning
* End-to-end
    * multi object segmentation problem 도 end-to-end 로 됨.
        * VOS 기법 중에 obj 종류만큼 inference 하는 애들도 있는데 이런 식으로 안 한다는 뜻
* String
    * object seg 잘 함

이를 만족하는 기법을 만들기 위해 FEELVOS 는 Pixel-Wise Metric Learning (PML) 에서 아이디어를 얻음. PML 은 pixel-wise embedding 을 triplet loss 을 이용하여 학습. test time 에는 first frame 과 nearst neighbor 로 predict

PML 은 fast 하고 simple 하지만 결과물이 별로임.

그래서 FEELVOS 는 PML과 비슷하게 learned embedding과 Nearst Neighbor 을 쓰되 그걸 final segmentation decision 에 사용하는 대신 internal guidance of the convolutional network 에 쓴다.

architecture 는 DeepLabv3+ 를 씀. 마지막 layer 없애고 embedding layer를 더해서 embedding feature vector 를 뽑음.

#### Methods

* Semantic Embedding
    * 각각의 픽셀 p 에 대해 semantic embedding vector $e_p$ in the learned embedding space 를 뽑는다.
    * 한 object 내의 pixel 은 embedding space 내에서 가깝고, 다른 object 끼리는 멀 것
    * 그런데 FEELVOS 에서는 이렇게 거리를 멀게 하도록 명시적으로 학습 시키지 않음 (PML 은 triplet loss 를 이용하여 명시적으로 함.)
    * FEELVOS 는 대신 dynamic segmentation head를 이용함.
    * distance between pixels p and q는 
    
    $$d(p, q)=1-\frac{2}{1+e^{\vert e_p - e_q \vert^2}}$$

* Global Matching
    * PML 과 비슷하게 first frame 의 semantric information 을 current frame 에서도 이용함.
    * embedding space 에서 nearest neighbor 를 이용하여 segmentation 함.
    * $P_t$ 를 time t에서의 모든 pixel 이라하고, $P_{t, o} \in P_t$ 를 time t에서 object o 에 속한 pixels 라고 하자.
    * 각각의 pixel $p \in P_t$ 그리고 정해진 o에 대해 $P_{1, o}$ 에 속하는 pixel 중 가장 가까운 distance 를 계산하여 global matching distance map $G_{t, o}(p)$ 를 얻는다.

    $$G_{t, o}(p) = \min_{q \in P_{1, o}} d(p, q)$$

    * **$P_{1, o}$ is not Empty.**

    * Global matching distance map 은 noisy 한 경향이 있다. 그래서 이를 바로 사용하지 않고 segmentation head 라는 것을 따로 만들어 noise 를 줄여줄 수 있도록 했다.

    * 구현은 large matrix product...

* Local Previous Frame Matching
    * first frame 의 semantic information 을 transfer 것(=global matching) 이외에도 FEELVOS는 전 frame의 information 을 transfer 하여 tracking 과 appearance change에 효과적으로 다룰 수 있게 했다.
    * Global Matching 과 비슷하게 $\hat{G}_{t, o}(p)$ 을 아래식과 같이 구함.

    $$\hat{G}_{t, o}(p) = \begin{cases} \min_{q \in P_{t-1, o}} d(p, q) & if P_{t-1, o} \ne \emptyset \\ 1 & otherwise \end{cases}$$

    * first frame 과 matching 할 때는 object가 많이 움직였을 수 있는데 previous frame 과 matching 할 때는 보통 조금만 움직이므로 False positive를 줄이고 computation time 을 줄일 수 있다. *(이 부분 이해 잘 안 됨. 왜 FP, computation time 줄어든다는 거지)*

    * 그런데 FEELVOS 는 사실 $\hat{G}_{t, o}(p)$ 안 씀. 대신 local matching distance map 사용한다.

    * local matching distance map 은 더 단순화 된 형태로 window 를 이용함. window size 를 k 라 하자. pixel p 에 대한 local Neighborhood $N(p)$ 를 p를 중심으로 x, y 축 방향으로 k 만큼 포함하는 $(2k+1)^2$ 개의 pixels 로 정의한다.

    $$L_{t, o}(p) = \begin{cases} \min_{q \in P_{t-1, o}^p} d(p, q) & if P_{t-1, o}^p \ne \emptyset \\ 1 & otherwise \end{cases} \ \ \ where\ P_{t-1, o}^p := P_{t-1, o} \cap N(p)$$

    * local matching distance map $L_{t, o}(p)$ 를 이용하는 게 computation cost 도 적고, 성능도 더 잘 나온다.

* Previous Frame Predictions
    * previous time frame 에 대한 prediction 을 local previous frame matching 에 쓰는 것 이외에도 object에 대한 posterior probability map 으로 사용함.

* Dynamic Segmentation Head
    * variable number of objects 을 체계적이고 효과적으로 다루기 위해 dynamic segmentation head 라는 것을 만듦.
    * inputs
        1. global matching distance map $G_{t, o}$
        2. local matching distance map $L_{t, o}$
        3. probability distribution for object o predicted at time t-1
    * output 은 각각의 object 에 대한 feature map
    * 각각의 object 에 대한 feature map 이 나오면 이를 각 pixel 별로 softmax 한뒤 cross entropy loss 이용

#### Training & Inference

* Training Procedure
    * 매 Training step 마다 mini-batch of videos 를 랜덤하게 고른다. 각 video 마다 우리는 랜덤하게 세 frame 을 고른다.
    * 세 frame
        1. reference frame(which plays role of the first frame of a video), 
        2. 연속된 프레임 두개. 하나는 previous 그리고
        3. 하나는 current frame
    * loss 는 current frame 에서만 구함.
    * training 할 때는 local matching distance map 을 구하기 위한 previous frame prediction 으로는 ground truth 사용함. 당연히 previous frame prediction 으로도 이거 씀
    * 끝. Simple!

* Inference
    * 위에 설명된 대로. 특별한 거 없음

#### Implementataion Details

* ImageNet / COCO pretrained DeepLabv3+ 이용
* bootstapped cross entropy loss 사용.
* 나머진 구현체 참조
