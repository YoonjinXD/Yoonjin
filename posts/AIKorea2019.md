#### 1. [산업체 세션2] SKT T-Brain, 김지원(SKT)

현재 SKT T-Brain 부서에서 진행해왔던 그리고 진행하고 있는 프로젝트들에 대해 소개하고 관련 이슈를 제시하는 세션이었습니다. SKT T-Brain 팀은 크게 4가지 이슈로 나뉘어 프로젝트를 진행하고 있습니다. 

1. Meta AI
   : 'AI가 AI를 만든다' 라는 슬로건을 걸고 딥러닝 모델 학습 자동화 플랫폼 시스템을 구축하는 팀으로,  Meta AI 인터페이스, 핵심 알고리즘 개발, 클러스터 인프라 구축 등으로 나뉘어 개발이 이루어지고 있습니다. 이 팀은 주어진 데이터를 학습하는 데 가장 효율적인 Neural Net과 그에 맞는 hyperparameter들을 자동적으로 탐색하여, 전문가나 사용자의 개입을 최소화함과 동시에 가장 적합한 AI 솔루션을 제공함을 주 목적으로 두고 있습니다. 현재 SKT T-brain 팀에서 가장 활발하게 개발되고 있다고 합니다.
2. Conversational AI
   : 자연어처리와 관련된 연구개발 팀으로, Question Answering, Dialogue Management 등의 연구를 통해 대화형 인공지능 개발(ex. NUGU) 에 주력하고 있습니다.
3. Visual AI
   : 컴퓨터비전과 관련된 연구개발 팀으로, Visual Question Answering, Visual Turing Test, Visual Relationship Detection 연구 등을 주력으로 하며 특히 자율주행 관련 테스크에 주목하고 있다고 합니다.
4. Music AI
   AI 스피커 '네모'를 통해 활용하려 하는 상용화 서비스로서 Singing Voice & Instrument Performance Translation 을 주로 연구하는데, 이 기술은 입력된 음악을 사용자가 원하는 목소리 및 악기 연주 스타일로 자동 변환해주는 기술입니다.

 

* 지난 프로젝트들
  * DiscoGAN
    
    기존의 GAN(Generative Adversarial Network) 모델들(DCGANs, ImprovedGan, cGan, pix2pix 등등)은 학습할 때 Paired Dataset, 즉 쌍이 매칭된 데이터셋이 필요하다는 한계점을 가지고 있었습니다. 쌍이 이루어진 데이터셋으로 학습을 하면 좋겠지만, 실제 세계에서 이러한 데이터를 찾기란 쉽지 않습니다. 이렇게 Supervised로서 제안된 모델을 Unsupervised 모델로  발전시켰다는 점에서 DiscoGAN(DiscoveryGAN)은 큰 의미를 가집니다. DiscoGAN은 기존 방식과 같이 Generator가 생성해낸 이미지를 Discriminator가 진위 여부를 판단하되, 해당 이미지를 다시 원래의 도메인으로 변환하는 또 다른 Generator를 배치하여 처음 인풋 이미지와 비교하는 Back Translation 기법을 사용합니다. 다시 말해, 인풋과 아웃풋간의 페어링이 되어 있지 않으니, 가짜 이미지 생성시 실제 인풋 이미지가 너무 많이 바뀌어 버리지 않도록 제재를 가하는 것이라 할 수 있습니다. 이 기법을 적용함으로써, DiscoGAN의 loss값은 Discrimination Loss(진위 여부 판별 loss) + Consistency Loss(다시 되돌아 온 생성된 이미지와 인풋 이미지 사이의 loss) 로 계산됩니다.
    이 논문은 CycleGAN이라는 유명한 모델과 매우 유사한 아이디어를 제안하는 논문으로, CycleGAN보다 조금 일찍 발표하여 당시 전세계의 이목을 끌었다고 합니다.  (그런데 CycleGAN 발표 때문에 조금 묻혔다고 합니다...)  기존 GAN의 한계점을 극복하고 특히 Cross-domain 영역에서 좋은 성과를 냈기에 GAN 역사상 유의미한 논문이라고 생각합니다.
    
    ![img](https://files.slack.com/files-pri/T25783BPY-F9RL21UGM/picture11.png?pub_secret=a59d54d636)
    
    [[논문]](https://arxiv.org/pdf/1703.05192.pdf) 
    
    [[코드]](https://github.com/SKTBrain/DiscoGAN)
    
    추가자료 - [CycleGAN 네이버D2 강연](https://www.youtube.com/watch?v=Fkqf3dS9Cqw&t=2799s)
    
  * Continual learning
    
    Continual Learning은 여러 task를 하나의 모델에 순차적으로 학습시켜, 최종적으로 모든 task의 수행이 가능한 모델을 학습시키는 기술입니다. 하지만 새로운 task를 학습 시킬 때, 모델이 새 task를 풀도록 완전히 바뀌어 이전 task를 모두 잊어버리는 Catastrophic Forgetting 현상이 발생하는데요, 이를 해결하기 위해 연구 팀은 GAN 기반 Generator와 주어진 task를 해결하는 solver의 두 모델로 구성된 Deep Generative Replay Framework를 제안하였습니다. Generator는 이전 모델의 Generator에서 생성된 가짜 데이터와 현재 task의 진짜 데이터를 합친 분포함수를 재현하도록 학습되고, 반면 solver는 그에 대한 적절한 타깃을 예측하도록 학습되는 구조입니다. 해당 연구는 'Continual Learning with Deep Generative Replay' 라는 제목으로 논문을 발표하여 지난 2017년 NIPS에 등재되었다고 합니다. 
    [[논문]](https://arxiv.org/pdf/1705.08690.pdf) (appeared in NIPS)
    [[코드]](https://github.com/kuc2477/pytorch-deep-generative-replay)
    
    
* 현재 진행되고 있는 프로젝트들
  * Auto Meta
    SKT T-brain팀은 Automatical Meta learning, 즉 자동적으로 메타 데이터를 학습할 수 있는 딥러닝 모델을 구축하고 hyperparmeter들까지 탐색해주는, 말 그대로 End-to-End AI 자동화 서비스 플랫폼을 개발하고 있습니다. 자동화된 도구로서 역할하는 것을 목표로, 다양한 딥러닝 관련 프로젝트에서 활용될 수 있으며 비전문가도 손쉽게 사용할 수 있도록 상용화할 계획이라고 합니다. Progressive NasNet에서 영감을 받아 착수한 프로젝트로 AutoML과 유사한 플랫폼이라고 볼 수 있습니다. 보여주신 데모 영상에서 그 완성도가 매우 높아보였는데, 아직 초기 개발 단계라고 하셔서 놀랐습니다. 상용화 전이라 자세한 내용은 설명해주지 못하셨지만 그냥 듣기에도 굉장히 크고 야심찬 프로젝트임을 알 수 있었습니다.
  * 자연어처리 분야에서는 BERT를 한국어에 조금 더 효과적으로 적용하기 위한 koBERT 프로젝트와 SUMBT(Slot-Utterance Matching for Universal and Scalable Belief Tracking) 프로젝트 등이 있습니다. SUMBT 는 올해 7월에 논문으로 제출되었다고 합니다. 관련 설명을 조금 해주셨는데, 지식이 짧아 이해하진 못했습니다. 시간이 되면 논문을 한번 읽어보고 싶습니다.
    [[논문]](https://arxiv.org/pdf/1907.07421.pdf)

우리나라 기업에서 진행하는 연구 개발이 생각보다 훨씬 더 수준급인 것 같아서 놀랍고 인상적이었던 세션이었습니다. 현재 연구하고 있는 주제들도 흥미진진하다고 생각했습니다. 하지만 Auto Meta와 같은 자동 모델 구축 및 최적화 알고리즘이 정말 필요한 일 일까 하는 의문이 들기도 했습니다. NasNet을 강하게 비판했던 [논문(Exploring Randomly Wired Neural Networks for Image Recognition)](https://arxiv.org/pdf/1904.01569.pdf)이 기억에 남는데, 여기서 주장하는 이야기가 아예 헛소리는 아니지 않을까...? 라는 생각이 들었습니다. 다른 분들은 어떻게 생각하는지 궁금하고 주제에 대해 이야기를 나누어보고 싶습니다. 



#### 2. [패턴인식/기계학습 여름학교 3] 최적화를 위한 반복법, 김동환(KAIST)

딥러닝의 최적화 메서드들에 대한 수학적 접근들을 주로 다루었던 세션입니다. 

* Optimization
  
  먼저, 최적화 문제란 어떤 목적함수의 함수값을 최적화(최대화 or 최소화)시키는 파라미터 조합을 탐색하는 문제를 말합니다. 목적 함수가 모든 파라미터에 대해 일차 이하의 다항식으로 구성되면 선형 최적화(linear optimization) 문제라 하고 그 외의 경우는 비선형 최적화(nonlinear optimization)이라 부릅니다.
  
* Iterative Method(반복법)
  이러한 최적화 문제를 푸는 원리는, 현재 위치에서 함수값이 감소하는 방향으로(최소화 문제라고 가정) 조금씩 파라미터 값을 이동해 나가는 것입니다. 이러한 이동을 더 이상 감소할 수 없는 곳(global minima)에 다다를 때까지 반복합니다. 이때 중요한 이슈는 어느 방향으로 갈 것인지, 얼만큼 이동할 것인지 결정하는 것입니다. 이 결정의 차이에 따라 다양한 최적화 기법이 나뉘어집니다.
  
  * Newton method
  * Gradient method
  * SGD
  * ADAM
  
* Convergence Rate(수렴속도)
  수렴 속도는 주어진 시간 내에 얼마나 빨리 원하는 해에 접근할 수 있는가에 대한 계산입니다. 조금 더 넓게 이야기하자면, 어떤 주제에 대한 적절한 답을 찾는데 어떤 알고리즘이 가장 빠르게 도달할 수 있는가에 대한 문제입니다.
  
  * Aysmptotic
  * Non-asymtotic * 
  
  * Sublinear / Linear / Superlinear rate
  
* Gradient Method
  * 사실 ADAM paper에서 Non-convex에서의 최적화 함수의 수렴 증명을 하지 못했습니다. 대신 Convex 스페이스에서 수렴함을 보이고 nonconvex 역시 수렴할 것이라는 믿음을 바탕으로 최적화 기법을 적용해보았는데 잘 되더라- 로 마무리 지었습니다. Convex optimization problems에 대한 증명은 [여기](https://wikidocs.net/17206) 있습니다. 
  * Non-convex 상황에서는 convex와는 달리 global minima의 존재 여부가 확실치 않고 여러개의 global minima가 나올 수 있기 때문에 그 존재를 정의하거나 탐색하기 매우 어렵습니다. 관련 연구 및 증명을 몇가지 설명해주셨는데... 어려웠습니다. Non-convex optimization problem에 대해 공부가 필요할 것 같습니다.
  * Smooth but nonconvex
    local minima의 존재에 의해 기존에 정의된 최적화 수식들이 의미가 없어질 뿐더러, Initial condition에 따라 다루려는 범위가 달라진다는 복잡한 문제들이 있습니다.
  
* Momentum method
  이런 문제들을 해소하고자 제안된 것이 Momentum method 입니다.
  
  * First-order method: 이전 누적된 gradient를 현재 계산에 사용하여 계산량을 늘리지 않고도 좋은 결과를 얻자는 아이디어. 일종의 관성 또는 가속도처럼 생각할 수 있다.
  * Heavy-Ball method: 위와 같은 아이디어를 살짝 개선시킨 방법입니다. ADAM은 요 수식의 알파베타를 matrix로 변환한 후 이것을 non-convex 공간에 적용한 모델이다.
  * Nesterov's method: 현재 위치에서 이전 속도만큼 일단 전진해보고 gradient를 계산한다. 선험적으로 혹은 모험적으로 에러를 교정해나간다고 할 수 있다. 
  * OGM: 이라는 게 있다고 한다.
  * nonconvex에서 Gradient를 어떻게 더 줄일까
  * SGD는 local minima를 벗어난다? -> 사실상 증명은 X -> saddle point 연구로 이어짐
  
* Saddle Point
  사실상 대부분의 딥러닝 모델들은 Non-convex 상황에서 학습되기 때문에 non-convex Gradient를 어떻게 더 줄일 수 있을까에 대한 연구는 꾸준히 진행되어 왔습니다. 특히, Global minima가 존재한다는 가정 하에 Non-convex 모델은 convex와 달리 local minima를 가질 수 있고, 그 local minima를 탈출해야 한다는 이론이 전반적으로 납득되어 왔었습니다. 하지만 2014년 이 문제에 대한 [논문](https://arxiv.org/pdf/1406.2572.pdf)이 하나 발표되었는데, 이 논문은 기존 상식을 깨고 local minima 문제가 사실은 고차원 공간에서는 발생하기 매우 희귀한 경우라고 주장했습니다. 어떤 critical point가 local minima가 되기 위해서는 모든 축의 방향으로 함수 그래프가 아래로 볼록해야 하는데, 축(차원)이 많아질수록 그 확률이 희박해지기 때문입니다. 논문에서는 Non-convex 상황에서 설령 local minima가 발생한다 하더라도, 그는 global minima와 거의 유사한 수준의 값이며, 게다가 실제 세계의 데이터셋은 non-convex 스페이스에 존재하므로 차라리 multiple global minima라 불러야 한다고 말입니다. 또한 이전에 모멘텀 기법이 성과를 보였던 것은 local minima를 탈출해서가 아니라, saddle point(안정점)을 탈출하기 때문이라고 합니다. 즉, non-convex 스페이스에서의 critical point들의 대부분은 saddle point 라는 주장을 하고 있는 것입니다. 이들은 위 주장과 함께, saddle point 탈출과 minima 근처에서의 수렴 속도 제어에 조금 더 초첨을 맞춘 saddle-free-newton-method 라는 새로운 최적화 알고리즘을 제안하였습니다. 아직 이들의 주장이 명확하게 증명되거나 반증된 것은 아니기 때문에 어떤 것이 사실일 지는 모르겠으나, 기존 '이렇게 해보니 잘 되더라-' 하는 식의 선험적인 연구만 반복되던 딥러닝 최적화 연구에 굉장히 유의미한 논문이 아니었을까 싶습니다.

  

이 세션을 듣고 말로만 듣던 Convex Optimization(볼록 최적화) 분야가 이런 거구나 알게 되었습니다. 지난 학기에 친구들이 볼록 최적화 강의를 들으며 고통 받는 것을 보았는데, 왜 고통 받았는 지 알 것 같기도 했습니다. 하지만 한편으로는 딥러닝 기저에 깔려있는 아이디어들이 모두 이러한 수학적 증명 및 논리로부터 시작되었음을 느낄 수 있었고 정말 흥미로운 내용들이 많다는 것을 알게 되었습니다. 항상 당연하게만 생각하던 딥러닝 메서드들에 대해 왜? 라는 반문을 할 수 있었고, 그런 반문에서부터 breakthrough가 일어났었음을 알게 되었습니다. 
이 세션은 개인적으로 가장 인상적인 세션이었습니다. 현재 딥러닝의 한계점 중 하나가 바로 어떠한 만족할만한 결과를 보고도 '왜 이런 결과가 도출되었는지 명확하고 합리적인 논증이 불가능함' 이라고 생각합니다. 학습 과정을 설명 또는 시각화 하기 위한 많은 노력 및 성과가 있었으나, 사실 수학적/논리적 증명만큼 깔끔하고 확실한 해석이 되지는 않습니다. 언제쯤 기계학습에 대한 명쾌한 해석이 가능해질지는 모르겠지만(100년 뒤..?), 이 세션을 통해 볼록 최적화 문제에 대해 한번 공부해보고 싶다는 생각을 하게 되었습니다.

(+) 최적화 공부에 좋은 자료를 추천 받았습니다 https://wikidocs.net/17052



#### 3. [패턴인식/기계학습 여름학교 5] 시계열 데이터와 기계학습 문제를 풀기 위한 신경망 구조, 정준영(Google DeepMind)

시계열 데이터(time-series data) 문제를 풀 수 있는 딥러닝 모델들을 주로 다룬 Introduction 세션입니다. 고전적 솔루션인 Kalman, Hidden Markov model부터 기본 RNN과 Long-term dependency 문제, 그 문제를 해결하기 위해 제안됐었던 LSTM, GRU 등 전체 흐름을 짚어보며 각 모델의 주요 아이디어를 recap 하는 시간이었습니다. 특별한 주제를 다룬 세션은 아니었지만, 최근 많이 들리는 XLNet 에 대해 관심 있었기 때문에 후반부 내용이 가장 흥미로웠습니다.  전반부(RNN ~ Transformer) 내용은 이전에 정리해둔 자료가 있으므로, 세션의 후반부(BERT, XLNet, Transformer XL)를 중심으로 리뷰하겠습니다. 

* BERT
  BERT 모델을 한 마디로 설명하자면, Attention is all you need 논문의 Transformer 중에서도 Encoder 블럭만 가져다가 여러개 붙여놓은 모델이라고 할 수 있습니다. 논문은 두가지 버전을 함께 소개하는데, 기본 BERT 모델인 BERT_base 는 12개의 인코더를 이어붙였고, 좋은 결과 수치를 얻기 위해 만든 BERT_large 는 24개의 인코더를 연결해놓았습니다. 뿐만 아니라 원래 논문보다 더 높은 차원의 Feed-forward Net을 배치하고 attention head를 더 많이 만들었고, Input sequence에 대한 embedding을 개선([WordPiece Embedding](https://arxiv.org/abs/1609.08144))시켰으며, Masked LM이라는 새로운 unsupervised training 알고리즘을 통해 양방향 context 분석을 가능케 하며 다양한 NLP task들을 성공적으로 수행하였습니다. Masked LM은 한 문장에서 랜덤으로 선택한 단어들을 마스킹한 후(구멍 뽁뽁 뚫기), 모델이 주변 단어들을 보고 마스킹 된(비어 있는) 단어들을 예측하는 식의 기법입니다. 
  전반적으로 BERT는 기존 Transformer의 메인 아이디어(사실 그냥 인코더)를 응용하되, 그 외의 부분을 모두 개조/발전 시켰다고 볼 수 있습니다. BERT는 공개된 후 기존 1등이었던 ELMO 모델을 밀어버리고 거의 대부분의 NLP task의 SOTA 자리에 앉게 되었습니다. (SOTA 수치들은 대부분 BERT_large 모델로 얻은 결과라고 합니다. 그래서인지 인코더를 무식하게 많이 쌓아올려 좋은 점수를 받은 것 뿐이라고 비판하는 리서처들도 꽤 있다고 합니다.) 

* Transformer XL
  XLNet을 설명하려면 Transformer XL에 대해 먼저 알아야 하는데, 이 모델은 XLNet 의 베이스가 된 모델입니다. BERT의 기초가 된 Transformer는 어텐션의 장점을 최대한 살리되 기존 RNN 방식을 완전히 탈피하고 주어진 시퀀스에 대한 동시 병렬 계산을 통해 Long-term dependency 문제를 해소시켜준 혁신적인 모델이었습니다. 하지만 이런 Transformer도 역시 한계를 가지고 있었는데, 시퀀스를 한번에 다 보기 때문에 시퀀스 길이가 고정적이어야 한다는 것, 그로 인해 긴 문장이 들어올 경우 문장을 잘라서(chunking) 판단해야 했고 고정 시퀀스 길이에 맞지 않는 나머지 부분은 버려야 했습니다. 이렇게 잘라진 문장 세그먼트들은 서로 간의 정보 전달이 이뤄지지 않았으므로, 너-무 긴 문장에 대해서는 아예 학습을 할 수 없었습니다. Transformer가 어느 정도 긴 문장에 대한 long-term dependency은 해결했지만... 고정 시퀀스 길이를 넘는 너무 긴 문장에 대한 학습은 어떻게 할까 하는문제의 솔루션으로서 제안된 것이 Transformer XL입니다. 아이디어는 바로 Transformer에 recurrence 추가하는 것입니다. 즉, 이전 스퀀스의 hidden state 값들을 caching하여 그들을 현재 시퀀스의 Key와 Vector값으로서 전달하자는 것입니다. 여기서 사용되는 것이 [relative positional encoding](https://arxiv.org/abs/1803.02155) 이며 관련 논문이 3~4편 정도 있다고 합니다. 이 모델에 대해 아주 잘 설명한 [블로그]( http://mlgalaxy.blogspot.com/2019/07/)가 있습니다.

* XLNet
  Transformer와 BERT가 NLP계의 Game Changer로 불리며 온갖 SOTA들을 휩쓸고 다닌지 1년도 채 안되었는데, 약 두달 전 XLNet 이라는 이름의 논문이 게재되면서 다시 한번 SOTA 자리가 갈아치워지고 있습니다. XLNet은 기존 NLP 모델들의 unsupervised representation learning 기법들(Autoregressive 이하 AR, Auto Encoding 이하 AE) 등을 대체할 새로운 기법으로 Permutation Language Modeling Obejective를 제안하였는데, 이를 통해 AR과 AE의 장점만을 살리고 단점은 극복할 수 있다고 설명합니다. 조금 더 설명하자면, 시퀀스 순서의 모든 permutation을 고려하고 모든 permuted factorization order들에서 모델 파라미터를 공유함으로써, BERT의 한계점이었던 독립성 가정과 pretrain-finetune  불일치 문제는 피하면서 bi-directional context는 유지할 수 있다고 합니다. (저도 무슨 소리인 지 잘 모르겠습니다) 동시에 Transformer-XL에서 제시된 recurrence 추가와 relative positional encoding을 활용하여 괄목할만한 성과를 냈습니다. 
  자세한 알고리즘은 논문을 공부해봐야 알 것 같습니다만, 강의자분께서 이 XLNet이 향후 1년간은 핫할 것이니 꼭 한번 읽어보라 권유하셨습니다. 동시에 BERT의 시대는 끝났다고 말씀하시기도 했습니다. BERT 공부 안하길 잘했습니다. ^0^/... 그래도 해야겠죠...?

* BERT -> LM (조경현님 논문)
  https://scholar.google.co.kr/citations?user=0RAmmIAAAAAJ&hl=en&oi=ao
  
  

NLP task solution들의 전체적 흐름을 정리하고 동시에 최근 활발히 연구되고 있는 주제들에 대한 전문가분의 의견을 들을 수 있어 정말 좋은 세션이었습니다. 특히 BERT와 XLNet에 대해 항상 궁금했었는데 전체적인 개요를 짚어주시고 향후 NLP 연구 방향에 대한 의견도 말씀해주셔서 좋았습니다. 앞으로 NLP 연구가 어떻게 진행될 지 기대됩니다.  