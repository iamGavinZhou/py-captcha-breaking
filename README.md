#说明
这是个破解验证码的完整示例，其中破解过程的**多处需要改进**，`+`代表需要改进的程度，主要有:
>> - CFS合并过程(`++`)
>> - 判断可能的字符数(`++++`)
>> - 字符分割(`++`)
>> - 单个字符识别精度(`+++`)

由于是业余时间写的，没放太多精力，有好的建议欢迎提出！或者联系作者：
>> **邮箱: gavinzhou_xd@163.com**

---
#破解流程
>> 1. `CFS(color fill segementation)`分割字符块
>> 2. 使用`cnn(convolution neural network)`判断字符块包含的字符个数
>> 3. 均分(`equally divide`)字符块，产生单个字符
>> 4. 使用cnn识别单个字符，获得最终答案

---


#各阶段精度
>> 1. CFS: ~= 99%
>> 2. 判断CFS块的可能字符数: ~= 80%
>> 3. 字符分割: ~= 98%
>> 4. 单字符识别: ~= 89% (with Caffe)

---
#训练各阶段源码
##判断长度
>> train_net/cnn_train_size*.py

##单字符识别
>> - train_net/cnn_train_chars.py
>> keras版本，效果不太好，accuracy大概84%
>> - data/lenet.prototxt
>> `caffe+mnist`识别，accuracy 89%左右,效果略好于keras版，**实际使用版本**

---

#Software requirements
>> - `Caffe`和`pycaffe`
>> 参考: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)
>> - `Keras` >= 0.2.x
>> 参考: [Keras installation instructions](http://keras.io/#installation)
>> - `cv2`(openCV的python接口)
>> 将`cv2.so`放入`/usr/lib/python2.7/dist-packages`
>> - `skimage`
>> - `numpy` and `pandas`
>> - `PIL`

---
#demo
>> ``` python
>> python demo.py | tee ./log/demo.log
>> ```

##demo accuracy
>> 由于多处存在瑕疵，**acuracy是`23/50 = 46%`**，效果不好，还需大力改进

##TODO
>> - Add a new implement
