# KMU UROP Program : MAML(Model-Agnostic Meta-Learning)
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
](https://arxiv.org/abs/1703.03400)

Meta-Learning의 방법론 중 Optimization-based Approach의 한 방법론인 MAML을 구현한다.

## Model
[Reference Code](https://github.com/dragen1860/MAML-Pytorch)

![image](https://user-images.githubusercontent.com/28241676/172050552-6433bc62-a80f-4ccd-b8ed-3a6e4d1855d6.png)

기존의 `nn.Module`를 이용하여 MAML의 inner loop와 outer loop를 구현

### Learner
`./learner.py`<br>
학습을 주도하는 모델의 구조를 처리하는 Class, 각 Layer에 대한 가중치 정보를 가지고 있어야 해서 다르게 정의됨
### Meta-Learner
`./meta_learner.py`<br>
MAML의 알고리즘(outer, inner loop)을 구현한 Model 구조 Class

## Tasks
1. Sinewave(Regression)
2. Omniglot(Classification)
3. Mini-ImageNet(Classification)
4. Stanford-Dogs(Classification)
### Regression
#### Sinewave
##### Dataset
+ `amplitude=1.5, phase=2.14 Sine-wave`
    + ![image](https://user-images.githubusercontent.com/28241676/172050910-797ac5e7-6548-46cc-a672-9d39d7fbd057.png)
+ Dataset Code : `./mydataset/sinewave.py`
##### Challenge
+ 여러 Amplitude와 Phase를 가지는 Sine-Wave의 좌표를 파악하는 Task
+ 즉, MAML를 통해 Sine-Wave의 본질적인 특성을 파악할 수 있는지에 대한 문제
##### Result
pass

### Classification
#### Omniglot
##### Dataset
+ [Download Link](https://github.com/brendenlake/omniglot)
+ ![image](https://user-images.githubusercontent.com/28241676/172051193-c2817fc0-1f9d-405c-a617-5729026b90de.png)
+ 50개의 언어 총 1623개의 글자로 이루어진 데이터
  + 50-Language, 1623-characters
  + 글자별 샘플은 20개를 사용(1623 x 23)
+ Dataset Code : `./mydataset/omniglot.py`
    + > omniglot : root<br>
    ____|- images/\*.png : All images <br>
    ____|- Source/\* : Source data <br>
    ____|- images.zip : images zip file <br>
    ____|- label.csv : label & tag <br>
    ____|- PreProcessing.ipynb : Pre-Procssing Code <br>
    ____|- train.csv <br>
    ____|- valid.csv <br>
    ____|- test.csv  <br>
+ 논문에서 제시한 Data Settings을 적용
    + Meta-Train 1200 characters
    + Meta-Valid 120 Characters
    + Meta-Test 303 Characters
      
##### Challenge
+ 각 언어별 글자를 파악하는 데, MAML를 적용
+ 이미지를 분류하는데, `글자(Language-Character)`라는 도메인을 전반적으로 이해 하고 있는 Model 구현
+ MAML를 통해 글자 도메인을 전반적으로 이해한 다음 새로운 Task(Image)에 대해 빠르게 적응(Fast Adaptation)하는 Model 설계

##### Result
+ Test-set Loss & Accuracy
  + ![omnigolot_loss_acc](https://user-images.githubusercontent.com/28241676/172055581-34237805-050d-4bc5-95e3-4efe8b811d5e.png)
  + 1 Epoch만에 Loss가 줄어들고 Accuracy가 늘어나는 것을 확인함

+ Performence Table
  + ![omniglot_performence_table](https://user-images.githubusercontent.com/28241676/172055583-2650e2bf-af9a-4a3f-9807-240e871e89b1.png)

#### Mini-ImageNet
##### Dataset
+ [Download Link](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4)
+ [train/val/test csv](https://github.com/twitter-research/meta-learning-lstm/tree/master/data/miniImagenet)
+ ![image](https://user-images.githubusercontent.com/28241676/172053722-54cb5d62-cea7-441a-bb30-ed31b2eef2d3.png)
+ 100개의 Classes, 600개의 Examples, 84x84 RGB Images
+ Dataset Code : `./mydataset/stanforddogs.py`
  + > ./stanforddogs/ : root<br>
  ____|- images/\*.png : All images <br>
  ____|- mini-imagesnet.zip : images zip file <br>
  ____|- train.csv <br>
  ____|- valid.csv <br>
  ____|- test.csv  <br>
+ 논문에서 제시한 Data Settings을 적용
    + Meta-Train 64 Classes
    + Meta-Valid 16 Classes
    + Meta-Test 20 Classes
      
##### Challenge
+ 사물 이미지를 파악하는 데, MAML를 적용
+ 이미지를 분류하는데, `현실의 물체(Real World Object)`라는 도메인을 전반적으로 이해 하고 있는 Model 구현
+ MAML를 통해 현실 세계의 물체에 대해 이해한 다음 새로운 물체 분류 TASK에 대해 빠르게 적응(Fast Adaptation)하는 Model 설계

##### Result
+ Test-set Loss & Accuracy
  + ![miniimagenet_loss_acc](https://user-images.githubusercontent.com/28241676/172055586-f904f157-ef2f-44b4-b506-b4d411b56664.png)
  + 1 Epoch만에 Loss가 줄어들고 Accuracy가 늘어나는 것을 확인함

+ Performence Table
  + ![minimagenet_performence_table](https://user-images.githubusercontent.com/28241676/172055587-99893128-d43d-4caf-b28c-633ff4ce07da.png)


#### Stanford-Dogs
##### Dataset
+ [Download Link](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
+ ![image](https://user-images.githubusercontent.com/28241676/172054758-c1891acf-f64f-430a-82bb-063c340fdaab.png)
+ 120개의 Classes, 20580개의 images
+ Dataset Code : `./mydataset/miniimagenet.py`
    + > ./miniimagenet/ : root<br>
    ____|- images/\*.png : All images <br>
    ____|- Source/\* : Source data <br>
    ____|- label.csv : label & tag <br>
    ____|- PreProcessing.ipynb : Pre-Procssing Code <br>
    ____|- train.csv <br>
    ____|- valid.csv <br>
    ____|- test.csv  <br>
+ 논문에서 제시한 Data Settings을 적용
    + Meta-Train 70 Classes
    + Meta-Valid 20 Classes
    + Meta-Test 30 Classes
+ Resizing 84x84 RGB Image
      
##### Challenge
+ 개 이미지를 파악하는 데, MAML를 적용
+ 이미지를 분류하는데, `개(Dogs)`라는 도메인을 전반적으로 이해 하고 있는 Model 구현
+ MAML를 통해 현실 세계의 강아지에 대해 이해한 다음 새로운 강아지 종을 분류하는 TASK에 대해 빠르게 적응(Fast Adaptation)하는 Model 설계

##### Result
+ Test-set Loss & Accuracy
  + ![stanforddogs_loss_acc](https://user-images.githubusercontent.com/28241676/172055593-c1277050-a47a-4eb4-8235-a5e7e8fa47c1.png)
  + 1 Epoch만에 Loss가 줄어들고 Accuracy가 늘어나는 것을 확인함

+ Performence Table
  + [Reference Paper 1](http://cs230.stanford.edu/projects_winter_2019/reports/15762310.pdf) : Datasets
    + ![image](https://user-images.githubusercontent.com/28241676/172055033-4932c19d-63ac-414a-8c67-876aaae60c82.png)
  + [Reference Paper 2](https://github.com/WenbinLee/CovaMNet) : CovaMNet
  + ![stanforddogs_performence_table](https://user-images.githubusercontent.com/28241676/172055594-73978042-e09f-4c78-b5fb-a57e339c45a0.png)
  