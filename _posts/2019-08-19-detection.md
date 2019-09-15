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

## Abstract

우리는 single neural network 가 bbox 와 class probabilities 를 전체 이미지에서 한번에 얻어낸다.  전체 detection pipeline 이 single network 이다. 훈련도 end-to-end 로 된다.

sota 모델들과 비교했을 때, YOLO 은 localization 은 약하지만 background 를 object 라 판단하는 false positive 는 별로 없다.

## Introduction

R-CNN 과 같은 기존 detection 들은 복잡하다. 우리는 object detection 을 single regression problem 으로 풀어볼 것이다. image 를 input 으로 받으면 bbox coord 와 각 bbox 의 class probability 를 예측한다. 이런 unified network 를 사용하면 You Only Look Once (YOLO) at an image 로 detection 해결 가능하다.

YOLO 는 부분 이미지를 보고 predict 하는 Fast R-CNN 와는 달리 전체 이미지를 보고 predict 하기 때문에 contextual information 을 쓸 수 있다. 이러한 contextual information 이 없는 Fast R-CNN 의 경우 background 를 object 라 판단하는 실수가 많은데 YOLO 은 그러한 실수를 줄였다.

또한 YOLO 는 보다 더 일반화 된 표현을 배울 수 있다. 무슨 말이냐면, domain shift 가 잘 된다는 뜻. (natural image 를 사용해서 train 한 모델이 artwork 에도 잘됨. R-CNN 과 그 격차가 큼.)

## Unified Detection

![](/assets/images/detection/yolo_v1_model.png)

우리는 object detection 의 seperate components 를 single neural network 로 unify 했다. 덕분에 end-to-end training 가능하고, 매우 빠르다.

우리는 먼저 input image 를 $S \times S$ grid 로 나눈다. obj 들은 각 중심이 어떤 grid 에 포함될텐데, 그 grid 가 obj 를 detect 하게 된다. (원문: If the center of an object falls in to a grid cell, that grid cell is responsible for detecting that object)

각 grid cell 는 B 개의 bbox 를 predict 하며, 더불어 confidence 와 class probability 도 예측한다. 이 confidence 는 그 bbox 가 obj 를 가지는지에 대한 모델의 확신도를 나타낸다. 우리는 confidence 를 $Pr(Object) * IOU_{pred}^{true}$ 로 정의했다.

각 bbox 는 5 개의 prediction (x, y, w, h, confidence) 로 구성된다. (x, y) 는 bbox 의 중심이 grid cell 의 중심에서 얼마나 벗어나있는지, (w, h) 는 whole image 에 비해 어느 정도 크기인지를 나타낸다. confidence 는 gt 와 predicted bbox 의 IOU 를 나타냄.

각 grid cell 는 C 개의 conditional class probability ($Pr(Class_i \vert Object)$) 를 예측한다. 

network prediction 의 크기는 $S \times S \times (B*5 + C)$ 가 된다.

PASCAL VOC 에서 우리는 S=7, B=2 를 사용했다. 

### Network Design

![](/assets/images/detection/yolo_v1_network.png)

우리 network 는 24개의 conv layer 과 2 개의 fc layer 로 구성됐다. 보다 더 빠른 Fast YOLO 로 만들었는데 이는 9개의 conv layer 를 사용했다.

### Training

먼저 앞 20개의 conv layer 는 ImageNet 으로 pretrain 함. (뒷부분은 avg pool + fc 등.., input size 는 224x224)

그 후 4개의 conv layer 와 2개의 fc layer 를 더 붙여서 detection training 함. Detection 는 좀 더 fine-grained visual information 을 필요로 하므로 이때는 input size 를 448x448 로 씀.

bbox 의 width height 는 원본 image 크기 대비 몇배인지 표현하도록 (0~1 로) normalize 함. (x, y) 는 grid cell 의 중앙으로부터 얼마나 떨어져있는지를 나타내며 grid cell 사이즈 대비 0~1 로 normalize 함. optimizer 로는 sum-squared error 가 optimize 하기 쉽다는 이유로 씀. 사실 이는 우리의 목적 average precision 과는 잘 맞지 않는 loss 임. 또 localization error 와 classification error weight 가 동일함. confidence 는 이미지의 대부분의 grid 가 obj 를 포함하지 않고 있기 때문에 훈련이 어려운 점이 있는데 이를 해결하기 위해서 두 개의 paramter ($\lambda_{coord} = 5, \lambda_{noobj}=0.5$) 를 만들었다. 

또한 sum-squared error 는 bbox 가 크던작던 상관없이 error 를 계산하는데 사실 bbox 가 클때 0.1 차이나는것과 작을때 0.1 차이는 서로 그 가중치가 다를 것이다. 이를 다루기 위해 우리는 width height 를 바로 쓰지 않고, 이를 root 씌운 값을 썼다.

training 할때는 한 obj 에는 하나의 bbox 만 responsible 하게 했다. 가장 큰 IOU 를 가진 bbox 를 responsible 하게 만들었다.

우리가 쓴 loss function 은 다음과 같다

$$\begin{matrix}
loss&=&&\lambda_{coord} \sum_{i=1}^{S^2} \sum_{j=1}^{B} \mathbb{1}_{ij}^{obj} [(x_i-\hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
&& +& \lambda_{coord} \sum_{i=1}^{S^2} \sum_{j=1}^B \mathbb{1}_{ij}^{obj} [(\sqrt{w_i}-\sqrt{\hat{w}_i})^2+(\sqrt{h_i}-\sqrt{\hat{h}_i})^2] \\
&& +& \sum_{i=1}^{S^2} \sum_{j=1}^B \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 \\
&& +& \lambda_{noobj} \sum_{i=1}^{S^2} \sum_{j=1}^B \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 \\
&&+&\sum_{i=1}^{S^2} \mathbb{1}_i^{obj} \sum_{c \in classes} (p_i(c)-\hat{p}_i(c))^2
\end{matrix}$$

보면 bbox coordi 와 confidence 는 그 bbox 가 responsible bbox 일때만 계산한다. class probability 는 그 grid 가 obj 를 가질때만 계산한다.

learning rate 어떻게 했고 몇 epoch 했고 등등은 논문에 나옴. dropout 도 씀.

### Inference

한 obj 에 여러 bbox 예측 되는 문제점 막기 위해 NMS 씀.

### Limitations of YOLO

한 grid 당 두개의 bbox 만 쓰고 하나의 class 만 예측할 수 있다는 spatial constraint 이 있다. 이런 spatial constraint 때문에 가까운 obj 는 predict 되지 않는다. 하지만 이 constraint 덕에 한 obj 에 여러 bbox 가 assign 되는 오류를 해결했다.

또한 aspect ratio 를 data set 을 통해 배우기 때문에 unusual 하거나 new aspect ratio 를 가진 obj 는 bbox regression 을 잘 못한다.

또한 multiple downsampling layer 를 거쳐 나온 feature 를 이용하여 bbox 를 예측하는데, 이는 너무 coarse 한 feature 라는 지적도 있다. 큰 obj 는 그래도 괜찮은데 작은 obj bbox regression 하기엔 너무 feature map의 크기가 작다. (그래서 ssd는 downsample 은 하되 중간 layer 의 output 도 씀)

또한 작은 bbox 와 큰 bbox의 regression 에서 크기에 대한 weight가 없다는 지적도 가능하다.

우리 모델의 error의 주요 원인은 incorrect localization 이다.

## Comparison to Other Detection Systems

pass

## Experiments

![](/assets/images/detection/yolo_v1_error.png)

localization error 가 높고, background 를 obj 라 판단하는 오류는 적다. Fast R-CNN 과 YOLO 을 결합하여 사용할 수 있는데 이렇게 할 경우 서로 단점을 보완해줘 성능이 크게 좋아진다.


![](/assets/images/detection/yolo_v1_general.png)

R-CNN 은 좋은 proposal 이 굉장히 중요한데, R-CNN 에서 사용하는 region proposal 인 selective search 가 natural image 에 fitting 된 알고리즘이다. 따라서 R-CNN 은 artwork 로 가면 안좋은 region proposal 을 받아 성능에 매우 큰 저하가 생긴다. 반면 우리 YOLO 는 generalization 도 잘된다.

## Conclusion

pass

# SSD (Single Shot MultiBox Detector)

## Abstract

SSD 는 bbox 들의 output space 를 각 feature map location 마다 여러 aspect ratio, scale 을 가지는 set of default boxes 로 규정함. (anchor box 이용한다.)

prediction 할 때는 각 anchor box 마다 presence score 와 refinement(x, y, w, h) 를 매김.

SSD 는 여러 feature map 들을 사용하여 다른 scale 의 obj 들을 잘 검출해낼 수 있음.

SSD 는 한번에 region proposal, classification 수행하기 때문에 빠르고, training 하기도 쉽다.

## 1. Introduction

우리는 bbox proposal stage 와 그에 따른 feature map resample stage 를 없앰으로써 속도를 높였다. 이는 이미 OverFeat, YOLO 에서 한 것이지만 우리는 작은 conv 를 이용하여 category score 를 매기는 것과 anchor 를 이용하여 bbox regression 을 하는 것, 그리고 multiple feature map 을 이용하는 것을 추가 함으로써 정확도를 상당히 높이는 기여를 했다. 이 세가지 개선 요소를 통해 mAP 를 63에서 74까지 끌어올릴 수 있었다. 

우리의 contribution 을 요약하자면 다음과 같다.

1. multiple categories 에 대한 single shot detector SSD 를 소개한다. 이는 YOLO 보다 훨씬 정확하다. 심지어 resion proposal 를 쓰는 매우 느린 Faster R-CNN 과 같은 모델 만큼이나 정확하다.
2. SSD의 core는 작은 conv filter 를 feature map 에 적용하여 category scores 와 box offsets 를 fixed set of default bounding boxes 에 대해 구한다는 것이다. 
3. 다른 scales 를 가지는 여러 feature map 을 통해 high detection accuracy 를 가질 수 있다.
4. 훈련도 쉽다.
5. 여러 데이터셋 PASCAL VOC, COCO 등에서 잘되는 걸 확인했다.

## 2. The Single Shot Detector (SSD)

Section 2.1 에서 SSD 의 구조를, Section 2.2 에서 훈련 방법을 설명한다. 다음 section (Sec.3) 에서는 dataset-specific model detail 를 설명한다.

### 2.1 Model

SSD 는 fixed-size collection of bounding boxes 와 그 box 에 속하는 obj 에 대한 score 를 예측하는 network 구조이다. 후에는 NMS 을 이용하여 최종 prediction 을 만든다.

**Multi-scale feature maps for detection**: 우리는 기존 network 의 끝을 자르고 feature map size 가 progressively 줄어드는 conv layers 를 붙였다. 그리고 이 layer 들의 feature map 들에 detection 을 했다. 이렇게 여러 scale 의 feature map 들을 이용하여 detection 을 수행했다. 각 feature map 마다 bbox 를 찾는 데 쓴 conv filter 는 서로 다르다.

**Convolutional predictors for detection**: 추가된 각 feature layer 는 fixed set of detection predictions 를 만들어낸다. 예를 들어 $m \times n \times p$ 의 feature map 이 있다고 하면 $3 \times 3 \times p$ 짜리 작은 conv filter 를 적용한다. 결과값으론 $m \times n$ location  마다 그에 해당하는 default bboxes 에 대한 bbox offset, category score 가 나온다. (YOLO 는 이 과정에 conv 를 사용하지 않고 feature map 을 Flatten 한 다음에 fc 를 썼었다.)

**Default boxes and and aspect ratio**: 각 feature map cell 은 default bboxes 를 가지고 있다. 이 default bboxes 마다 offset 과 category 를 predict 한다. 각 cell 마다 k개의 default bboxes 를 가진다 하면 각 cell 의 output 은 (# of classes + 4)k 가 될 것이다. 이는 Faster RCNN 의 anchor 와 매우 유사한 개념이다.

### 2.2 Training

**Matching  strategy**: gt box 가 어떤 default box 와 매치하는 지 결정하는 것. 우리는 각 gt box 와 가장 큰 IoU 를 가지는 default box 를 true example 로 함. 그리고 또한 gt box 와 IoU 가 0.5 가 넘는 모든 default box 들을 true example 로 함.

**Training Objective**: Multibox 의 loss 와 비슷함.

$$L = \frac{1}{N}(L_{conf}(x, c) + \alpha L_{loc}(x, l, g))$$

N은 matched default boxes 의 갯수. (N=0 이면 그냥 loss=0 으로 함) $L_{loc}$ 은 faster R-CNN 과 같음 (smooth L1). $\alpha$ = 1 로 함.

$x_{ij}^p={1, 0}$ 를 i-th default box 가 class p 를 가지는 j-th gt box 에 매칭될 때 1이 되는 indicator 라 하자.

$$L_{loc} (x, l, g) = \sum_{i \in Pos}^{N} \sum_{m \in \{cx, cy, w, h\}} x_{ij}^k smooth_{L_1}(l_i^m-\hat{g}_j^m)$$

$$L_{conf}(x, c) = -\sum_{i \in Pos}^N x_{ij}^p \log(\hat{c}_{i}^p) - \sum_{i \in Neg} \log(\hat{c}_i^0)$$

**Choosing scales and aspect ratios for default boxes**: 우리는 lower & upper feature maps 를 모두 이용하여 detection 을 수행한다. 다른 level 의 feature map 들은 서로 다른 receptive field size 를 가진다. 하지만 운좋게도 ssd 의 default box 는 actual receptive field 에 꼭 연관될 필요는 없다. 그래서 우리는 각 layer의 default box 들이 우리가 정해주는 특정한 scale 을 배우도록 할 수 있었다.

우리가 정한 layer 별 default boxes 의 scale 은 다음과 같다.

$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1} (k - 1), k \in [1, m]$$

$s_{min}$ 은 0.2 즉 lowest layer 는 0.2 의 scale 을, $s_{max}$ 는 0.9 즉 highest layer 는 0.9의 scale 을 배우도록 했다.

그리고 aspect ratio 는 $\{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$ 을 썼다. aspect ratio 가 1 일때는 scale 이 $s_k' = \sqrt{s_ks_{k+1}}$ 인 default box 를 추가했다.  

**Hard Negative mining**: 대부분의 default box 들이 negative 이다. 이는 심각한 불균형을 초래함. 그래서 우리는 모든 negative box 를 사용하지 않고 conf loss 가 가장 큰 것들만 뽑아서 훈련에 사용했다. positive bbox 와 negative bbox 의 비율이 1:3 이 될때까지 뽑음. 이를 통해 빠르고 안정적인 학습이 가능했다.

**Data Augmentation**: pass

## 3. Experimental Results

base network 로는 VGG16 사용했다. 조금 변형함.

## 3.1 PASCAL VOC2007

잘된다. 왜 잘되냐면 SSD 는 여러 category 의 obj 들을 잘 잡는다. 또한 two decoupled step 을 통해 obj 을 localization 하지 않고 directly 학습하기 때문에 localization 을 잘한다. 그러나 SSD는 비슷한 obj category 는 많이 혼란해하는제 이는 우리가 multiple category 에 location 을 share 하기 때문으로 보인다. 

## 3.2 Model analysis

SSD를 더 잘 이해해보자.

**Data augmentation is crucial**: data aug 를 하냐 안하냐에 따라 모델 성능이 10% 가까이 바뀜.(mAP 74.3 $\rightarrow$ 65.5)

**More default box shapes is better**: aspect ratio 를 여러개 하는게 중요하다.

**Atrous is better**: atrous 사용하자.

**Multiple output layers at different resolutions is better**: SSD 의 가장 큰 contribution 은 여러 output layers 에서 여러 scale 의 default box 를 뽑아냈다는 점이다. 

<!-- $$\begin{tabular}{l|ccccccc}
model        & pascal voc 07 & pascal voc 10 & pascal voc 12 & pascal voc 07+12 & mscoco & gpu   & cpu \\
R-CNN        & 66.0          & 62.9          & 62.4          & 62.4             &        & 47    &     \\
FAST R-CNN   & 70.0          & 68.8          & 65.7          & 68.4             & 19.7   & 0.3+2 &     \\
FASTER R-CNN & 73.2          &               &               &                  &        & 0.13  &     \\
YOLO v1      & 63.4          &               &               &                  &        & 0.025 &    
\end{tabular}$$
 -->
<!-- 
$$\begin{array} {|r|r|}\hline 0_0 \\ \hline  \end{array}$$ -->