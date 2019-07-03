---
layout: post
title: "convert model from pytorch to tensorflow"
categories: ETC
author: lee gunjun
---

# pytorch 에서 학습된 모델 tensorflow 로 변환하기

----

pytorch 로 개발을 하는 개발자도 가끔 모바일 배포등의 이유로 tf 로 모델을 변환해야 할 필요가 생길때가 있다. pytoch 모델을 tf 모델로 변환하는 방법에 알아보자.

다음과 같은 순서대로 하면 된다.

1. pytorch 모델을 똑같이 tensorflow 로 만든다.
2. 이름을 동일하게 해준다.
3. th->tf로 학습된 parameter를 옮겨준다.

다른 방법으론 onnx 를 쓰는 게 있다.
onnx가 예전엔 지원되는 operations가 적고 변환된 결과물도 난잡하여 별로였지만 최근엔 크게 발전해서 쓸만해졌다. 여전히 지원 안 되는 op는 있지만 우리가 쓰는 웬만한 op는 웬만해선 변환이 잘 되고 여전히 변환된 모델이 난잡하고 속도 측면에서도 비효율적으로 변환되는 문제가 있지만 그렇게 크지 않다.

추천하는 작업 프로세스로는 demo 에서는 onnx 을 쓰고 배포할 때는 tf 모델을 직접 짜서 param을 옮긴 모델을 쓰는 것이다.

onnx 로 변환하는 법도 아래에 적어놨다.

## 1. pytorch 모델을 똑같이 tensorflow로 만들기 & 2. 이름(scope) 동일하게 해주기.

```
class ThNet(nn.Module):
    def __init__(self):
        super(ThNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv2 = nn.Conv2d(2, 4, 5, padding=2)
        self.fc1 = nn.Linear(4*2*2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 4*2*2)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
```

와 같은 pytorch 모델이 있다고 하자.

이를 똑같이 tensorflow 로 짜준다. 그리고 이름도 맞춰준다. (tf의 scope을 열심히 쓰면 된다.)

**경고: pytorch의 view와 tensorflow의 reshape의 방식이 다르다**  
이를 해결하기 위해 transpose 를 사용했다.

```
import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense, flatten

def TfNet(x):
    x = conv2d(x, 2, [5, 5], [1, 1], padding='same', name='conv1', activation=tf.nn.relu)
    x = max_pooling2d(x, [2, 2], [2, 2], padding='valid')
    x = conv2d(x, 4, [5, 5], [1, 1], padding='same', name='conv2', activation=tf.nn.relu)
    x = max_pooling2d(x, [2, 2], [2, 2], padding='valid')
    x = tf.transpose(x, perm=[3, 0, 1, 2])
    x = tf.reshape(x, [-1, 16])
    x = dense(x, 1024, activation=tf.nn.relu, name='fc1')
    x = dense(x, 10, name='fc2')
    return x
```

## 3. th->tf로 학습된 parameter를 옮기기


```
th_model = ThNet().eval()

tf_input = tf.placeholder(tf.float32, [None,8,8,1])
tf_model = TfNet(tf_input)

saver = tf.train.Saver()

m = {}
for k, v in th_model.named_parameters():
    m[k] = v

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('logs', sess.graph)
    
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        torch_name = v.name.replace('weights:0', 'weight') \
                            .replace('kernel:0', 'weight') \
                            .replace('biases:0', 'bias') \
                            .replace('bias:0', 'bias') \
                            .replace('norm:0', 'norm.weight') \
                            .replace('/', '.')
        print('transform:', v.name, v.get_shape(), torch_name, m[torch_name].size())

        if len(v.get_shape()) == 4:
            sess.run(v.assign(m[torch_name].permute(2, 3, 1, 0).data.numpy()))
        elif len(v.get_shape()) == 1:
            sess.run(v.assign(m[torch_name].data.numpy()))
        elif len(v.get_shape()) == 2:
            sess.run(v.assign(m[torch_name].data.numpy().T))
        else:
            raise Exception('unknown shape')
    
    # 저장
    tf.train.write_graph(sess.graph, './trans/', 'squeeze_graph.pbtxt')
    save_path = saver.save(sess, "./trans/squeeze.ckpt")

    print('transfer done')
    writer.close()
```

위 코드는 fc layer, conv layer와 batch normalization layer 만 고려하여 짠 코드다.  
추가적으로 원하는 layer가 있다면 v.name.replace(...) 를 적절하게 추가해서 써주자.

위 코드까지 실행하는데 성공했다면 ckpt 와 pbtxt 을 성공적으로 얻을 수 있다.

# 추가

* pbtxt와 ckpt을 이용하여 pb 파일 생성
* pb 파일 tensorflow에서 쓰기
* pb 파일을 tflite 로 변환
* pb 파일을 coreml 로 변환
* pytorch 를 onnx 로 변환
* onnx 를 pb 로 변환

## pbtxt와 ckpt을 이용하여 pb 파일 생성

----

tensorflow 의 freeze_graph를 이용하여 pbtxt와 ckpt를 이용하여 pb 파일을 만들 수 있다.  
freeze_graph는 tf를 설치할 때 자동으로 같이 설치된다.

```
freeze_graph --input_graph={pbtxt 경로} \
             --input_checkpoint={ckpt 경로} \
             --output_graph={pb 파일 출력 경로} \
             --output_node_names={tf의 출력 node 이름} 
```

```
ex) freeze_graph --input_graph=trans/squeeze_graph.pbtxt \
                 --input_checkpoint=trans/squeeze.ckpt \
                 --output_graph=trans/squeeze.pb \
                 --output_node_names=Softmax
```

output_node_names는 출력 node의 이름을 넣어주면 된다.  
출력 node의 이름은 tensorboard 켜서 확인하는 게 제일 안전하고 쉽다.


## pb 파일 tensorflow 에서 쓰기

----

```
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

        return graph

if __name__ == '__main__':
    graph = load_graph('{pb file path}')
    for op in graph.get_operations():
        print(op.name)
    x = graph.get_tensor_by_name('{input tensor name}')
    y = graph.get_tensor_by_name('{output tensor name}')

    with tf.Session(graph=graph) as sess:
        out = sess.run(y, feed_dict={x:np.random.randn(1, 3, 224, 224)})
        print(out)
```

graph를 로드하고 (load_graph)  
input tensor와 output tensor를 찾아서 (graph.get_tensor_by_name)  
sess.run 해주면 된다.

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

    input_tensor = sess.graph.get_tensor_by_name('{input tensor name}')
    output_tensor = sess.graph.get_tensor_by_name('{output tensor name}')
    tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [input_tensor], [output_tensor])
    open("{tflite 파일 저장할 경로}", "wb").write(tflite_model)
```
operation이 아니고 tensor를 찾아야 하므로 get_tensor_by_name 에 :0을 넣어주는 걸 잊지 말자.

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

# onnx

*주의: onnx 는 trace-based exporter(export 할 때 trace를 이용, 한번 executing 해봐야 함) 라서 dummy input 의 크기나, batch-size 가 달라지면 잘 작동 안 할 수 있음.*

## pytorch 를 onnx 로 변환

----

install onnx: conda install -c conda-forge onnx

```
import torch
import torchvision

model = torchvision.models.squeezenet1_1()
model.load_state_dict(torch.load('input.pth'))
torch.onnx.export(model, torch.ones(1, 3, 224, 224), 'output.onnx') #(model, dummy input, save path)
```

onnx model 를 inspecting
```
import onnx
model = onnx.load('output.onnx')

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
```

## onnx 를 pb 로 변환

----

### installation

#### PIP
```
pip install onnx-tf 
```

#### Git
```
git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow
pip install -e .
```

### Do it!

```
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("input.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # import the onnx model to tf

print(tf_rep.inputs)
print('-----------')
print(tf_rep.outputs)
print('-----------')
print(tf_rep.tensor_dict)

# inference
output = tf_rep.run(np.random.randn(1,3,224,224))._0
# output.shape == (1, 1000)

tf_rep.export_graph("output.pb")  # export the model
# input tensor: 'import/'+tf_rep.tensor_dict[tf_rep.inputs[0]].name
# output tensor: 'import/'+tf_rep.tensor_dict[tf_rep.outputs[0]].name
```

onnx 을 이용하여 pb 을 생성하면 input tensor 와 output tensor 의 이름을 tf_rep.inputs 와 tf_rep.outputs 로 쉽게 찾을 수 있다. 그렇게 찾은 tensor 이름 앞에 import/ 를 붙여주면 변환된 pb에서의 tensor 이름이 된다.