---
layout: post
title: "detection"
categories: CNN
author: lee gunjun
---




# R-CNN

3 step (+2 step) 으로 이루어짐
- region proposal (selective search 이용)
- CNN 을 이용한 fixed length feature extraction
- SVM 을 이용하여 image classification

+ 추가 2-step
    - SVM을 통해 각 class 별로 score가 매겨진 bounding boxes들이 나오는데 이를 greedy non-maximum suppression을 통해 필요한 것만 간추려냄.
    - bounding box을 정답에 더 가깝게 살짝 조정해주는 Bounding-box regression 실행. CNN이 positional invariant 하기 때문에 이런 과정으로 정확도 올릴 수 있음

R-CNN은 region proposal method에 agnostic 함. 현존하는 region proposal methods 중에서 아무거나 갖다 써도 됨. 논문에선 selective search 사용.

region proposal에서 2000개의 proposals를 만들고 이를 전부 fixed size(227x227 RGB)로 crop-resize 함.

CNN은 imagenet pretrained model(VGG 혹은 AlexNet) 을 이용함.

pretrained model을 fine tuning 하기 위해 각 class 별로 ground-truth box와 IOU가 0.5가 넘으면 positive sample로 그렇지 않으면 negative sample로 이용.

SVM을 학습할때는 ground-truth box을 positive sample로, 0.3보다 작은 걸 negative sample로 이용함

CNN을 학습할 때와 SVM을 학습할 때 positive & negative sample을 뽑는 방법이 다른 이유는 그렇게 해보니 더 잘됐기 때문

CNN 학습하고, SVM학습하고 Bounding box regressor 학습하고 총 3번 학습함.

----

**greedy non-maximum suppression**: 결과로 [Pc, bx, by, bx, bx] 가 나오면 , Pc < 0.6인건 다 제거. 남은 것중 Pc 가 가장 큰걸 선택 선택된 박스와 IOU가 0.5인 박스를 모두 제거. 이를 계속 반복

# FAST R-CNN

## Introduction

R-CNN 보다 training time 은 9x, inference time 은 213x 빠르다.

Region-based Convolutional Network method (R-CNN) 는 좋지만 아래의 단점을 가진다.

1. Training is a multi-stage pipeline
    - CNN, SVM, bbox regressor 3번 훈련
2. Training is expensive in space and time
    - SVM 학습할 때 feature 들은 모두 각 obj proposal 마다 추출되어 저장되어야 함. 5k images of VOC07 trainval set 에 이걸 하려면 2.5 GPU-days 걸림. 디스크는 몇백 기가 차지함.
3. Object detection is slow
    - 느림. VGG16 쓰면 47s/image

모든 object proposal 에 대해 ConvNet forward 를 하는데, 부분적으로 겹치는 애들은 computation 을 share 하게끔 하면 빠르게 할 수 있지 않을까? -> 실제로 SPPNet 은 share 하게끔 하여 10x 가량 빠름.

그러나 SPP 는 여전히 multi-stage pipeline (CNN, SVM, bbox regressor 3번 훈련) 을 가짐. 

이러한 문제를 해결함. Fast R-CNN 은

1. Higher detection quality (mAP)
2. Training is single-stage, using multi-task loss
3. Training can update all network layers
4. No dist storage is required for feature caching

## 2. Fast R-CNN architecture and training

Fast R-CNN 의 구조

![](/assets/images/detection/fast_r-cnn.png)

1. 전체 image을 several convs 에 집어넣어 conv feature map 을 얻는다.
2. feature map 으로부터 각 RoI 마다 fixed-length feature vector 를 얻는다.
3. 각 feature vector 를 fc layers 에 넣어 두 가지 output 을 한번에 얻는다.
    - 하나는 backgound 를 포합한 K+1 개 class 에 대한 확률값.
    - 또 다른 하나는 4 values encodes refined bbox positions for one of the K classes

이를 위해선 RoI Pooling layer 가 필요함. RoI pooling layer 은 max pooling 을 이용하여 RoI 안의 feature 를 고정된 사이즈 HxW 의 feature map 으로 변환해주는 것.

cnn 은 pretrained model (VGG16) 가져옴.

Fine tuning 과정에 대해 알아보자.

SPPnet 은 다른 image 에서 온 training sample 을 back-prop 하는게 매우 비효율적이라 spatial pyramid pooling layer 하에서 weight 을 update 하는게 불가능했다. 이러한 비효율성은 각 RoI 가 굉장히 큰 receptive field 를 가지기 때문.

반면 우린 feature sharing 을 하기 때문에 괜찮다. 먼저 N 개의 images 을 뽑고 R/N 개의 RoI 를 각 image 에서 뽑아 R개의 RoI 를 얻어낸다. 한 image 에서 나온 RoI 는 같은 computation, memory 을 공유함. 이럴 경우 한 image 에서 나온 RoI 들은 correlated 되어 있어 training 이 잘 안될 거라 걱정할 수 있지만 practical 하게 잘됨. 논문 저자는 N=2, R=128 사용했다.

loss 는 two sibling output layers 를 같이 학습시키기 위해 Multi-task loss 를 사용함. 

$$L(p, u, t^u, v) = L_{cls} (p, u) + \lambda [u \ge 1] L_{loc} (t^u, v)$$

$$L_{cls}(p, u) = -\log p_u$$

$$L_{loc}(t^u, v) = \sum_{i \in \{x, y, w, h\}} smooth_{L_1} (t_i^u - v_i)$$

$$smooth_{L_1}(x) = \begin{cases} 0.5x^2 & if \left\vert x \right\vert \lt 1 \\
\left\vert x \right\vert - 0.5 & otherwise \end{cases}$$

p 는 class prediction u 는 class 의 index 를 가리킴. backgound 는 u=0. t 는 bbox regression

smooth L1 는 outlier 에 less sensitive 함. 

실험에선 $\lambda=1$ 씀.

우리는 training 때 64 개의 RoI 를 (N=2, R=128) 각 image 에서 뽑아쓴다. 우리는 gt bbox 와 IoU 가 0.5 이상 되는 RoI 중 25% 를 뽑는다. 64개 중 남은 RoI 는 IoU 가 [0.1, 0.5) 에 속하는 것들 중 높은 것들부터 뽑아 채운다. 이는 background example 이 된다. 즉 u=0

object detection 이 scale-invariant 하도록 하기 위한 방법으론 두가지가 있다.

1. brute force learning
    - 각 image 는 predefined size 로 resize 되어 training, testing 됨. network 가 알아서 scale invariant 하게 학습되길 기도함. 짧은 쪽 pixel (=s)가 600pixel 되도록 함. 단 긴쪽이 1000넘어가면 안됨. aspect 는 original image 그대로 유지.
2. using image pyramids
    - image pyramids 를 통해 scale invariant 하게 만들어줌.

## 3. Fast R-CNN detection

network 는 한개의 image 혹은 하나의 image pyramid 와 R 개의 object proposals 를 input 으로 받음. (test time 에서 R는 보통 2000 정도됨.) R-CNN 처럼 nms 씀.

detection 을 빠르게 하기 위해 Truncated SVD 씀. 보통의 image classification 에선 fc 가 conv 에 비해 훨씬 적은 계산량을 차지하지만, detection 에서는 RoIs 가 많으면 거의 절반의 시간을 fc 에 쓴다. 따라서 fc 를 빠르게 하기 위해 truncated SVD 를 도입했다.

![](/assets/images/detection/truncated_SVD.png)

## 4. Result

잘됨

![](/assets/images/detection/scale_invariance.png)

우리의 기도가 잘 통하여 scale 을 하나로만 해도 꽤 scale invariant 한 것을 알 수 있음. 그래서 최종 모델은 그냥 scale 하나만 써서 함.

svm 을 써도 acc 엔 별 차이 없음.

항상 proposal 이 많을 수록 좋은가? -> no. 너무 많으면 오히려 나빠지기도 함.

# Faster R-CNN

Fast R-CNN 이나 SPPnet 에서 매우 큰 발전을 이루었지만 여전히 region proposal 은 bottleneck 임. 우리는 이를 해결할 Region Proposal Network (RPN) 을 제시함. RPN 은 nearly cost-free region proposal 을 제공함. 

## 1. Introduction

Fast R-CNN 은 region proposal 단계에서 매우 큰 시간이 소요된다. 현재 대부분의 region proposal 들은 CPU 에서 implemented 되어 있어 GPU 의 이점을 누리지 못한다. 속도를 높이기 위해 이를 GPU 로 re-implementation 하는 방법이 있겠으나 우리는 새로운 네트워크 (RPN) 를 제안하여 속도를 높여볼 것이다.

우리는 region-based model 들의 convolutional feature map 을 region proposal 에 사용할 수 있다는 것을 발견했다.   
우리는 이 conv feature 위에 RPNs(two conv layers) 를 추가하여 region proposal 를 얻어냈다.

training 은 multi-stage 이지만 simple 함.

## 3. Region Proposal Networks

![](/assets/images/detection/faster_r-cnn.png)

RPN 은 any size 의 image 를 input 으로 받아 region proposals 와 그의 objectness score 을 출력하는 것. backbone CNN network 로는 ZFnet 와 VGG 을 사용했다.

RPN 의 구성

1. Conv2D(256, kernel_size=3) 을 이용하여 각 pixel 마다 256-d 의 vector 를 뽑아냄.
    - 각 pixel 마다 크기 3x3의 sliding window 하는 것과 동일함.
2. Conv2D(2k, kernel_size=1) 을 이용하여 objectness score 를, Conv2D(4k, kernel_size=1) 을 이용하여 bbox regression 을 예측함.
    - 논문에서 k는 9를 씀.

### Translation-Invariant Anchors

각 sliding window 마다 우린 k개의 region proposals 를 얻는다. 이 각 sliding windows 의 k개 reference box 를 Anchors 라 부른다. 각 Anchor 중심은 그 sliding windows 에 위치한다. 우리는 각 sliding windows 마다 3개의 scale 과 3개의 aspect ratio, 총 9개의 anchors 를 사용했다. 이 접근의 중요한 property 는 이 방법이 translation invariant 하다는 점이다.

### A Loss Function for Learning Region Proposals

각 anchor 는 binary class label (사실 binary 가 아니라 3개 부류임. positive, negative, not assigned 중 하나) 을 갖는다. 어떤 gt box 가장 큰 IoU 를 갖는 anchor 박스이거나 0.7보다 큰 IoU 를 갖는 gt box 가 있으면 positive 를 준다. 이렇게 하면 한 gt box 는 최소 1개의 anchor 와 매칭 된다. positive 가 아니면서 IoU 가 0.3 이상인 gt box 가 없는 anchor 는 negative 가 된다. positive도, negative 도 아닌 것은 loss 계산할 때 빠진다.

$$L(\{p_i\}, \{t_i\})=\frac{1}{N_{cls}} \sum_{i} L_{cls}(p_i, p_i^\ast)+\lambda \frac{1}{N_{reg}} \sum_i p_i^\ast L_{reg} (t_i, t_i^\ast) $$

i 는 anchor 의 index 이다. $L_{cls}$ 는 log loss, $L_{reg}$ 는 smooth L1 이다. reg loss 는 positive anchor 에서만 계산된다.

우리는 regression 을 위해 4개의 coordinates 를 다음과 같이 parameterize 했다.

$$t_x = (x-x_a)/w_a, t_y=(y-y_a)/h_a, t_w=log(w/w_a), t_h=log(h/h_a)$$

$$t_x^\ast = (x^\ast-x_a)/w_a, t_y^\ast=(y^\ast-y_a)/h_a, t_w^\ast=log(w^\ast/w_a), t_h^\ast=log(h^\ast/h_a)$$

$x, x_a, x^\ast$ 는 각각 predicted box, anchor box, gt box 이다.

### Optimization

우리는 single image 에서 많은 positive 와 negative anchor 를 추출하여 이를 mini-batch 로 이용했다. 모든 anchor 를 이용하면 negative anchor 가 dominate 하기 때문에 positive 와 negative anchor 의 수를 1:1 로 하여 256개의 anchors 를 sample 해냈다. positive 가 128개가 안되면 그만큼 negative 를 더 넣었다.

CNN backbone 은 ImageNet pretrained model 을 사용했다.

### Sharing Convolutional Features for Region Proposal and Object Detection

detection (=classifier) 으로 우리는 Fast R-CNN 을 사용한다. 이제 RPN 과 Fast R-CNN 이 공유하는 conv layers 를 어떻게 training 할지 알아보자.

RPN 과 Fast R-CNN 은 독립적으로 training 된다. Fast R-CNN 은 region proposal 은 학습되지 않으므로 one-stage 훈련이 쉽게 됐다. 하지만 Faster R-CNN 은 RPN 과 Fast R-CNN 이 훈련중에 서로 어떤 영향을 줄지 모르므로 훈련을 조심히 해야한다. 우리의 training stage 는 총 4개로 구성된다.

1. RPN 을 학습. ImageNet pretrained model 을 가져와 fine tuning 함
2. RPN 에서 나온 proposals 들을 이용하여 detection network 를 학습시킴. 이 detection network 는 imagenet pretrained model 을 이용. 즉 이 때는 RPN과 detection network 가 conv layers 을 공유하지 않음.
3. RPN 의 backbone 을 detection network 의 backbone 으로 변경하고 RPN 만의 conv layer (중간, cls, reg) 을 학습함. 이제 RPN 과 detection 은 conv layer 공유함.
4. detection 만의 fc layer 학습

### Implementation Details

짧은 변의 길이 s가 600이 되도록 resize 함.
single scale 은 multi-scale 에 비해 성능은 좀 낮지만 빠름.

anchor 는 3개의 scale ($128^2, 256^2, 512^2$) 과 3개의 aspect ratio (1:1, 1:2, 2:1 ) 씀

image boundary 를 넘어가는 anchor box 들은 training 과정에서 안씀. 그래서 전체 anchor box가 20k 정도인데 ($\approx 60 \times 40 \times 9$) cross-boundary anchor 빼면 6k 정도만 남았다.

nms 이용함. IoU 0.7 기준으로 다시 redundancy 줄임. 한 이미지에 anchor는 2k 정도가 남음.

## 4. Experiment

RPN 과 detection 이 conv layers 을 share 할때가 unshare 할 때 보다 성능이 좋다. pascal voc 2007 에서 mAP 1% 가량 차이남. (58.7% $\rightarrow$ 59.9%)

RPN 을 쓰는게 ss 같은 거 쓰는 것보다 성능이 좋다.

# YOLO v1










\begin{table}[]
\begin{tabular}{l|ccccccc}
model & \begin{tabular}[c]{@{}c@{}}pascal voc 07\\ (mAP)\end{tabular} & \begin{tabular}[c]{@{}c@{}}pascal voc 10\\ (mAP)\end{tabular} & \begin{tabular}[c]{@{}c@{}}pascal voc 12\\ (mAP)\end{tabular} & pascal voc 07+12 & mscoco & gpu & cpu \\
R-CNN & 66.0 & 62.9 & 62.4 & 62.4 &  & 47 &  \\
FAST R-CNN & 70.0 & 68.8 & 65.7 & 68.4 & 19.7 & 0.3 &  \\
FASTER R-CNN &  &  &  &  &  &  & 
\end{tabular}
\end{table}