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

----

### 1. 랜덤 없애기

```
random.seed(1000) # random 을 쓰는 경우
np.random.seed(1000) # np.random 을 쓰는 경우
torch.manual_seed(1000) # torch cpu
torch.cuda.manual_seed_all(1000) # torch gpu
torch.backends.cudnn.deterministic = True # cudnn 을 쓰는 경우
```
