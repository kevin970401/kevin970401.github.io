---
layout: post
title: "Pytorch Tips"
categories: Pytorch
author: lee gunjun
---

# Pytorch Tips

----

## 목차

1. 랜덤 없애기
2. tensorboardx

----

### 1. 랜덤 없애기

지난 실험을 재현하고 싶을 때 random 때문에 되지 않는 경우가 많다. 특히 cudnn을 사용하는 경우 그 자체의 random 성 때문에 더 힘들다. 이런 random 을 없애는 방법을 알아보자

```
random.seed(1000) # random 을 쓰는 경우
np.random.seed(1000) # np.random 을 쓰는 경우
torch.manual_seed(1000) # torch cpu
torch.cuda.manual_seed_all(1000) # torch gpu
torch.backends.cudnn.deterministic = True # cudnn 을 쓰는 경우
```

### 2. tensorboardx

tensorboard 는 머신러닝을 학습시키는 과정에서 매우 편리하게 사용할 수 있는 모니터링 도구일 뿐더러 매우 좋은 시각화 도구기도하다. tensorflow 에는 tensorboard 지원이 잘 되는데 pytorch 에서는 잘 지원이 되지 않아 아쉬움이 큰데 tensorboardx 를 설치하면 pytorch 에서도 tensorboard 를 어느정도 사용할 수 있다.

설치:

```설치
pip install tensorboardx
```

example:

```example
from tensorboardX import SummaryWriter

writer = SummaryWriter('logs/first-train') # logs/first-train 폴더에 기록하는 writer를 만든다
writer.add_scalar('tag/name', value, step) # scalar를 기록한다.
```
