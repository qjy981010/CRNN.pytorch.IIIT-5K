For Chinese: 详细做法请参考[我的博客](https://qjy981010.github.io/2017/12/24/PyTorch-%E7%94%A8CRNN%E6%94%BB%E9%99%B7IIIT-5k/)

## RCNN
A pytorch implementation of CRNN，and test it with IIIT-5K.

Paper is [here](https://arxiv.org/abs/1507.05717).

Netword Struct：

| Type | Configurations | Output Size |
| :---: | :---: | :---: |
| Input | W × 32 gray-scale image | W × 32 × 1 |
| Convolution | #maps:64, k:3 × 3, s:1, p:1 | W × 32 × 64 |
| MaxPooling | Window:2 × 2, s:2 | W/2 × 16 × 64 |
| Convolution | #maps:128, k:3 × 3, s:1, p:1 | W/2 × 16 × 128 |
| MaxPooling | Window:2 × 2, s:2 | W/4 × 8 × 128 |
| Convolution | #maps:256, k:3 × 3, s:1, p:1 | W/4 × 8 × 256 |
| Convolution | #maps:256, k:3 × 3, s:1, p:1 | W/4 × 8 × 256 |
| MaxPooling | Window:1 × 2, s:2 | W/4 × 4 × 256 |
| Convolution | #maps:512, k:3 × 3, s:1, p:1 | W/4 × 4 × 512 |
| BatchNormalization | - | W/4 × 4 × 512 |
| Convolution | #maps:512, k:3 × 3, s:1, p:1 | W/4 × 4 × 512 |
| BatchNormalization | - | W/4 × 4 × 512 |
| MaxPooling | Window:1 × 2, s:2 | W/4 × 2 × 512 |
| Convolution | #maps:512, k:2 × 2, s:1, p:0 | W/4-1 × 1 × 512 |
| Map-to-Sequence | - | W/4-1 × 512 |
| Bidirectional-LSTM | #hidden units:256 | W/4-1 × 256 |
| Bidirectional-LSTM | #hidden units:256 | W/4-1 × label_num |
| Transcription | - | str |

Please install [warp-ctc for pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding) first.
click [here](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) and download IIIT-5K dataset to the 'data/' folder of current path.

