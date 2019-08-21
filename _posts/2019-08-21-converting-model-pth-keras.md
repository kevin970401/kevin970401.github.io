---
layout: post
title: "convert model from pytorch to keras"
categories: ETC
author: lee gunjun
---

# pytorch 에서 학습된 모델 keras 로 변환하기 + keras 모델 tflite 변환하기

##  pytorch 에서 학습된 모델 keras 로 변환하기

pytorch 에서 학습한 모델을 keras 로 옮기는 방법에 대해 알아보자.

간단하게는 onnx 을 쓰는 방법이 있으나 onnx 로 변환된 모델은 비효율적으로 동작하는 경우가 많다. 순수한 Conv, Pool, fc 로 이뤄진 모델은 몇% 느려지는 정도일 수 있으나, DepthwiseConv 같은 연산을 쓰는 경우 심하게는 몇배씩 느려진다.

추천하는 방법은 pytorch 모델과 같은 형태의 keras 모델을 직접 짜고 parameter 를 옮겨주는 것이다.

아래 pth2keras() 는 서로 이름이 동일한 layer 를 찾아 pth model (=pytorch model) 의 parameter 들을 keras model 로 옮겨준다.

```
def pth2keras(pth_model, keras_model):
    m = {} # m : {'classifier.1.bias': np.array(...), ...}

    for k, v in pth_model.named_parameters():
        m[k] = v
    for k, v in pth_model.named_buffers(): # for batchnormalization
        m[k] = v

    with torch.no_grad():
        for layer in keras_model.layers:
            if isinstance(layer, DepthwiseConv2D):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].permute(2, 3, 0, 1).data.numpy()) # weight
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # bias

                layer.set_weights(weights)
            elif isinstance(layer, Conv2D):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].permute(2, 3, 1, 0).data.numpy()) # weight
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # bias

                layer.set_weights(weights)
            elif isinstance(layer, BatchNormalization):
                print(layer.name)
                weights = []
                if layer.scale:
                    weights.append(m[layer.name+'.weight'].data.numpy()) # gamma
                if layer.center:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # beta
                weights.append(m[layer.name+'.running_mean'].data.numpy()) # running_mean
                weights.append(m[layer.name+'.running_var'].data.numpy()) # running_var
                layer.set_weights(weights)
            
            elif isinstance(layer, Dense):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].t().data.numpy())
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy())
                layer.set_weights(weights)
```

예시로 pytorch model과 keras model 을 간단하게 만들어 봤다.

```
import torch
import torch.nn as nn
from tensorflow.keras.layers import (
    Conv2D, Dense, BatchNormalization, 
    DepthwiseConv2D, Input, 
)
from tensorflow.keras.models import Model

class PthModel(nn.Sequential):
    def __init__(self):
        super(PthModel, self).__init__(nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False))

pth_model = PthModel()
inputs = Input(shape=[224, 224, 3])
outputs = Conv2D(1, kernel_size=3, strides=1, padding='same', use_bias=False, name='0')(inputs)
keras_model = Model(inputs=inputs, outputs=outputs)
```

pth_model 과 keras_model 은 둘다 하나의 Conv2D 로 만들어진 동일한 형태의 모델이다.

pth_model 의 parameter 를 keras_model 로 옮기는 것은 pth2keras 함수를 통해 간단하게 할 수 있다.

```
pth2keras(pth_model=pth_model, keras_model=keras_model)
```

자! 완성이다!

실제로 parameter 가 성공적으로 옮겨갔는지 임의의 input 을 통해 확인해보자.

참고로 pytorch 와 keras의 input 값, output 값의 shape가 다르므로 이를 주의해주자.

```
import numpy as np
pth_inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32)
keras_inputs = pth_inputs.permute(0, 2, 3, 1).numpy()

pth_outputs = pth_model(pth_inputs)
pth_outputs = pth_outputs.permute(0, 2, 3, 1).data.numpy() # keras output 과 형태가 맞도록 변형
keras_outputs = keras_model.predict(keras_inputs)
print(np.abs(pth_outputs-keras_outputs).max())
```

출력값이 0이 나온 것을 통해 결과값이 동일하게 나온 것을 확인할 수 있다! (어느 정도 오차로 인해 0이 아닐 수 있다. <1e-5)

모델을 저장하고 싶다면 save_model 을 이용하자. hdf5 형식으로 저장해준다.

```
from tensorflow.keras.models import save_model
save_model(keras_model, 'path/to/h5file.h5')
```

## keras 모델 tflite 변환하기

hdf5 형식으로 저장된 keras 모델을 tflite 로 변환하는 것은 매우 간단하다
아래 코드대로 실행하면 된다.

```
import tensorflow as tf
h5_path = 'path/to/h5file.h5'
converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
tflite_model = converter.convert()
open('path/to/tflite_file.tflite', 'wb').write(tflite_model)
```

완성!

혹은 weight 를 uint8 로 quantize 하고 싶다면 아래와 같이 하면 된다.

```
converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_weight_quant_model = converter.convert()
open('path/to/tflite_weight_quant_model.tflite', 'wb').write(tflite_weight_quant_model)
```

완성!

혹은 weight 와 activation 모두 uint8 로 quantize 하고 싶다면 아래와 같이 하면 된다. activation 을 quantize 하기 위해선 소량의 데이터셋(=representative_dataset)이 필요하다.

만약 모델 input 의 size 가 (224, 224, 3) 이 아니라면 아래 representative_dataset 을 그에 맞춰 바꿔주자

```
def representative_dataset():
    with tf.compat.v1.Session() as sess:
        imgs = ['path/to/img1', 'path/to/img2', ...]
        for img_path in imgs:
            img = Image.open(img_path).resize((224, 224)).convert('RGB')
            img = np.asarray(img).astype(np.float32).reshape((1, 224, 224, 3))
            img = img / 256
            yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_full_quant_model = converter.convert()
open('path/to/tflite_full_quant_model.tflite', 'wb').write(tflite_full_quant_model)
```

완성!

변환된 tflite 모델을 다시 실행해보고 싶다면 [tflite interpreter](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python) 를 사용하면 된다.

전체코드는 [이 곳](https://www.dropbox.com/s/30ix3w8vwk2oe9z/convert.py?dl=0) 에서 다운받을 수 있다.
