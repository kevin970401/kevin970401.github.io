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

Fast R-CNN 의 구조

![](/assets/images/detection/faster_r-cnn.png)

전체 image을 

# YOLO v1

---|----|---