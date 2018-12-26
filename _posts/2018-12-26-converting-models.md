---
layout: post
title: "convert model from pytorch to tensorflow"
categories: ETC
author: lee gunjun
---

# pytorch 에서 학습된 모델 tensorflow 로 변환하기

----

다음과 같은 순서대로 하면 된다.

1. pytorch 모델을 똑같이 tensorflow 로 만든다.
2. 이름을 동일하게 해준다.
3. th->tf로 학습된 parameter를 옮겨준다.

onnx 을 쓰는 방법도 있지만 지원 op 도 적고 결과물이 복잡하고.. 등등의 이유로 안 썼다.

onnx 로 변환하는 법도 아래에 적어놨다.

## 1. pytorch 모델을 똑같이 tensorflow로 만들기 & 2. 이름(scope) 동일하게 해주기.

```
SqueezeNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (3): Fire(
      (squeeze): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (4): Fire(
      (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (6): Fire(
      (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (7): Fire(
      (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (8): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (9): Fire(
      (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (10): Fire(
      (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (11): Fire(
      (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
    (12): Fire(
      (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

와 같은 pytorch 모델이 있다고 하자. (위 모델은 [Cadene의 pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) 의 squeezenet1_1 을 마지막 부분만 살짝 바꾼 것)

이를 똑같이 tensorflow 로 짜준다. 그리고 이름도 맞춰준다. (tf의 scope을 열심히 쓰면 된다..) 

```
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

def fire_module(inputs, squeeze_depth, expand_depth, scope=None):
    with tf.variable_scope(scope) as fire_scope:
        net = _squeeze(inputs, squeeze_depth, scope=fire_scope)
        net = _expand(net, expand_depth, scope=fire_scope)
        return net

def _squeeze(inputs, num_outputs, scope):
    squeeze_conv = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')
    return squeeze_conv

def _expand(inputs, num_outputs, scope):
    e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='expand1x1')
    e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='expand3x3')
    return tf.concat([e1x1, e3x3], 3)


class Squeezenet(object):
    def __init__(self, num_class):
        self._num_classes = num_class

    def build(self, x):
        self._is_built = True
        return self._squeezenet(x)

    def _squeezenet(self, images):
        with tf.variable_scope('features'):
            net = conv2d(images, 64, [3, 3], stride=[2, 2], scope='0')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='maxpool1', padding='VALID')

            net = fire_module(net, 16, 64, scope='3')
            net = fire_module(net, 16, 64, scope='4')

            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='maxpool4', padding='VALID')

            net = fire_module(net, 32, 128, scope='6')
            net = fire_module(net, 32, 128, scope='7')
            
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='maxpool8', padding='VALID')

            net = fire_module(net, 48, 192, scope='9')
            net = fire_module(net, 48, 192, scope='10')
            net = fire_module(net, 64, 256, scope='11')
            net = fire_module(net, 64, 256, scope='12')

        net = conv2d(net, self._num_classes, [1, 1], stride=[1, 1], scope='last_conv')
        net = tf.nn.avg_pool(net, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], name='avgpool10', padding='VALID')
        net = tf.squeeze(net, 1)
        net = tf.squeeze(net, 1)
        logits = tf.nn.softmax(net, axis=-1)
        return logits
```

## 3. th->tf로 학습된 parameter를 옮기기


```
th_model = pretrainedmodels.squeezenet1_1(num_classes=1000) # pytorch의 squeezenet
th_model.last_conv =  nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

th_model.load_state_dict(torch.load('trans/squeeze.pth')) # 학습된 parameter load

init_img = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name='input')
tf_model = Squeezenet(2)
out = tf_model.build(init_img) # tf 모델 생성

saver = tf.train.Saver()
m = {}
for (k, v) in th_model.named_parameters():
    m[k] = v

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)

    # 학습된 parameter 전달
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):

        torch_name = v.name.replace('weights:0', 'weight') \
                            .replace('biases:0', 'bias') \
                            .replace('norm:0', 'norm.weight') \
                            .replace('/', '.')

        print(v.name, v.get_shape(), torch_name, m[torch_name].size())

        if len(v.get_shape()) == 4:
            v.load(m[torch_name].permute(2, 3, 1, 0).data.numpy())
        elif len(v.get_shape()) == 1:
            v.load(m[torch_name].data.numpy())
        else:
            raise Exception('unknown shape')
    
    # 저장
    tf.train.write_graph(sess.graph, './trans/', 'squeeze_graph.pbtxt')
    save_path = saver.save(sess, "./trans/squeeze.ckpt")

    print('transfer done')
    writer.close()
```

위 코드는 아마 conv layer와 batch normalization layer 만 고려하여 짠 코드일 것이다 \
추가적으로 원하는 layer가 있다면 v.name.replace(...) 를 적절하게 추가해서 써주자.

위 코드까지 실행하는데 성공했다면 ckpt 와 pbtxt 을 성공적으로 얻을 수 있다.

아래는 추가다.

# 추가

* pbtxt와 ckpt을 이용하여 pb 파일 생성
* pb 파일을 tflite 로 변환
* pb 파일을 coreml 로 변환
* pytorch 를 onnx 로 변환
* onnx 를 pb 로 변환

## pbtxt와 ckpt을 이용하여 pb 파일 생성

----

pb 파일은 tensorflow 의 freeze_graph를 이용하여 pbtxt와 ckpt를 이용하여 만들면 된다. \
freeze_graph는 tf를 설치할 때 자동으로 같이 설치된다.

```
freeze_graph --input_graph={pbtxt 경로} \
             --input_checkpoint={ckpt 경로} \
             --output_graph={pb 파일 출력 경로} \
             --output_node_names={tf의 출력 layer 이름} 
```

```
ex) freeze_graph --input_graph=trans/squeeze_graph.pbtxt \
                 --input_checkpoint=trans/squeeze.ckpt \
                 --output_graph=trans/squeeze.pb \
                 --output_node_names=Softmax
```

output_node_names는 출력 node의 이름을 넣어주면 된다. \
출력 node의 이름은 tensorboard 켜서 확인하는 게 제일 안전하고 쉽다.


## pb 파일을 tflite 로 변환

----

```
import tensorflow as tf


with tf.Session(graph=tf.Graph()) as sess:
    with tf.gfile.GFile("{pb 파일 경로}", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(graph_def, name='')
    
    for op in sess.graph.get_operations():
        print(op.name)

    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name('Softmax:0')
    tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [input_tensor], [output_tensor])
    open("{tflite 파일 저장할 경로}", "wb").write(tflite_model)
```
operation이 아니고 tensor를 찾아야 하므로 name 에 :0을 넣어주는 걸 잊지 말자.

위 코드가 실행됐다면 tflite 파일이 생성된다.


## pb 파일을 coreml 로 변환

----

먼저 pip install tfcoreml 로 tfcoreml을 설치하자

```
import tfcoreml as tf_converter

tf_converter.convert(
    tf_model_path = '{pb 파일 경로}',
    mlmodel_path = '{mlmodel 파일 저장할 경로}',
    output_feature_names = ['Softmax:0'], # Softmax:0 대신 원하는 텐서로 바꾸자
    input_name_shape_dict = {'input:0': [1, 224, 224, 3]} # input:0 대신 원하는 텐서로 바꾸자
)
```

[tfcoreml](https://github.com/tf-coreml/tf-coreml) 을 이용하면 쉽게 된다.

## pytorch 를 onnx 로 변환

----

**틀린 정보 있을 수 있음**

```
import torch
import torchvision

model = torchvision.models.squeezenet1_1()
model.load_state_dict(torch.load('input.pth'))
model.features[2].ceil_model = False # pooling layer 의 ceil mode가 True 면 onnx 로 변환이 안 됨.
model.features[5].ceil_model = False # pooling layer 의 ceil mode가 True 면 onnx 로 변환이 안 됨.
model.features[8].ceil_model = False # pooling layer 의 ceil mode가 True 면 onnx 로 변환이 안 됨.
torch.onnx.export(model, torch.ones(1, 3, 224, 224), 'output.onnx')
```

## onnx 를 pb 로 변환

----

**틀린 정보 있을 수 있음**

먼저 pip install onnx-tf 를 한다.

```
import onnx
from onnx_tf.backend import prepare


onnx_model = onnx.load("input.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation

tf_rep.export_graph("output.pb")  # export the model
```
