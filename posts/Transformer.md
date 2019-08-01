### Transformer

앞서 RNN의 첫등장 -> 문장이 길어질수록 앞의 내용을 잊어버리는 Long-term dependency 문제 -> 그로 인해 고안되었던 attention 기법,  LSTM, 그리고 GRU 라는 성과들에도 불구하고, 이들은 모두 순차적으로 정보를 읽어들이는 RNN 기반 구조였기에 Long-term dependency를 완전히 탈출하지는 못했다. 이러한 한계점을 타개해준 논문이 바로 그 유명한 'Attention is all you need' 이다. 어텐션이 어텐션 받고 있는 만큼 딥러닝에 관심 있다면 반드시 한번쯤 읽어보는 것이 좋다. 이 논문은 제목 그대로, 기존 RNN 기반 모델들을 아예 사용하지 않고 attention 기법만을 적용해도 sequence 모델을 다루는 데 손색이 없고 심지어 더 좋은 성능과 결과를 보인다는 내용이다. 

주요 아이디어는 대략 이렇다. 문장에서 중요한 부분만 기억하며 순차적으로 읽어나가는 것이 아니라, 병렬화(Parallelization)을 통해 한 방에 문장을 위에서 관조하자! 전체 문장에서 주목해야할 부분에 가중치를 높이는 방식이기 때문에 어떠한 정보를 잊어버릴 일이 없다. 즉, Long-term dependency를 완벽히 극복~~(비록 또다른 문제가 발생하긴 했지만)~~해낸 것이다. 이 논문에서 제시한 모델이 바로 이름부터 멋있는**Transformer** 이고 이것은 여러 분야에 응용되며 다양한 분야의 state-of-the-art 자리에 앉게 되었다.
(+) 이 논문 이후 해당 아이디어를 응용한 모델들이 마구 나와서, 사실 원조 Transformer 자체로는 SOTA 자리를 오래 차지하지 못했다고 한다. ~~완전 춘추전국시대~~



#### Overview

![img](/home/jenzzz/Documents/Yoonjin/posts/images/transformer.png)

*[Figure 1] Transformer architecture -ref [paper](https://arxiv.org/pdf/1706.03762.pdf)*

Transformer의 기본 구조도이다. 왼쪽 회색 박스가 encoder, 오른쪽 박스가 decoder이며, 생전 처음 보는 것들이 Positional Encoding, Multi-Head Attention, Masked Multi-Head Attention, shifted right 등등으로, 그냥 보기만 해도 어려워보인다.



#### Modules

1. **Multi-Head Attention**
   먼저 Transformer에서 가장 핵심적인 메커니즘인 Multi-head attention 모듈이 있다. 이를 위해 논문은 <u>Self-attention</u> 알고리즘을 고안해냈다. 

   * Self Attention

     ![img](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

     *[Figure] Self-attention 에 대한 직관을 잘 보여주는 그림 - ref [jay alammar's blog](http://jalammar.github.io)*

     예시 문장에서 it 이라는 단어를  해석하기 위해서는 앞에서 언급되었던 the animal 이라는 단어에 주목하듯이, self-attention 메커니즘에선 각 단어마다 자신과 가장 연관있는 데이터들에 대한, 즉 자신의 attention 정보를 벡터의 형태로 가지고 있다. 이걸 어떻게 할까? 논문에서는 이 attention 연산을 'Scaled dot-product attention' 이라 표현한다. 

   * Scaled Dot Product Attention

     ![scaled-dot-product-attention](https://pozalabs.github.io/assets/images/sdpa.PNG)

     ![1562661109138](/home/jenzzz/Documents/Yoonjin/posts/images/1562661109138.png)

     Q는 query 벡터를, K는 key 벡터를, V는 Value 벡터를 의미한다. dk는 key 벡터의 디멘션을 의미한다. 여기서 요 벡터들이 갑자기 왜 나와서 이러는 지 혼란스러운데... 이 친구들은 그냥 self attention을 구현하기 위해 필요한 몇가지 역할들을 각각 담당하는 벡터들이라고 생각하면 된다. 

     **(필요한 몇가지 역할?)** Query 벡터는 말 그대로 현재 내가 보고 있는 단어를 뜻하고, 여기에 첫번째로 곱해지는 Key 벡터는 어떤 단어와의 상관 관계 정보를 담고 있다고 생각할 수 있다. 이 둘을 matmul 했을 때 나오는 결과를 Score라고 부르는데, 이 값이 높으면 두 단어의 연관성이 높음을 뜻한다(and vice versa).

     ![1563258013135](/home/jenzzz/Documents/Yoonjin/posts/images/1563258013135.png)

     이후 이 스코어 값을 0과 1사이의 확률값으로 표현하기 위해 softmax function을 취해주는데 이 때 Key 벡터의 루트값으로 먼저 score를 나눠준다. 논문에서는 key 벡터의 차원이 커질수록 스코어 계산값이 증대하는 문제를 보완하기 위해 추가한 연산이라고 한다. 

     여튼 이렇게 계산된 확률 값들을 Value 벡터와 다시 곱해준다. 이 Value 벡터는 앞서 나온 Key 벡터와 pair 즉 한 쌍을 이루고 있는 것으로 **현재 내가 이해하기론 기존 워드 임베딩과 같은 존재 같다.** 따라서, Softmax값을 Value 벡터와 곱하는 이유는 내가 현재 보고 있는 이 단어와 가장 유의미한 관계를 가지고 있는 단어 임베딩을 가장 선명하게, 별 상관없는 단어의 임베딩은 아주 희미하게 남기겠다는 뜻이다. 마지막으로 요 값들을 모두 시그마한  것이 Attention layer를 거친 어떤 단어 A의 최종 output이 된다. 이는 문장 속의 어떤 단어 A가 가진 의미, 중요성, 즉 우리가 기존 attention 기법에서 사용한 hidden layer와 비슷한 시맨틱이라고 보면 된다.

     ![1563258472548](/home/jenzzz/Documents/Yoonjin/posts/images/1563258472548.png)

     하지만 이해될 때만 이해되고 또 뒤돌아 보면 이해가 안될 수 있으니 마지막으로 한번만 더 정리해보자. Q와 K를 Dot product한 후 루트 dk로 스케일링하는 과정에 의해 둘 사이의 유사도를 구한 뒤(이 과정은 cosine similarity와 유사) 이 유사도를 전체 문장에 대해 softmax를 취해 확률값으로 나타낸다. 이 값을 Value 벡터와 dot-product 함으로써 query 와 유사한 value일수록, 중요한 value일 수록 결과 값이 높아진다. 즉, 현재 단어의 output에 그 단어와 가장 연관 높은 단어들의  Value 벡터들을 가장 많이 포함시키려는 과정이다.(softmax 연산을 왜 집어넣었는 지 알 것 같다.)

   이제 Multi-head attention 중에 'attention' 까지는 알겠다. 남은 'Multi-head'는 그래도 비교적 간단한데, 위와 같은 scaled-dot product 연산을 여러개의 쓰레드(=head)로 내려 한 단어에 대해 여러번 attention 정보를 얻어내겠다는 것이다. 예를 들어, 만약 head가 4개라면 4개의 attention 벡터가 생성되고 이 벡터들을 concat하여 최종 attention 벡터를 도출하게 된다. 따라서 각 head의 attention 벡터 차원 곱하기 head 개수는 반드시 모델의 차원이 되어야 한다.

   여기 그 수식 그림

   근데 이런 일을 왜 할까? 문장 요소 중 어떤 단어들은 2~3개의 단어와 밀접한 연관을 가지고 있을 수 있고, 또는 모호한 애매한 관계를 가지고 있을 수 있다. 이런 단어 간의 연관성을 보다 정확하게 짚어내기 위해, 여러 관점에서 attention 연산을 수행하기 위함이다. Convolutional Network에서 하나의 filter가 여러번 feature map을 뽑아내는 것과 비슷하다.

   

2. **Feed Forward**

   

3. **Positional Encoding**

4. **Encoder and Decoder**

   Encoder Layer

   - Positional Encoding 이 Back prop때 vanishing 되는 현상을 방지하기 위해 Residual Block 기법을 사용하여 Positional 정보를 전달.
   - Multi-head Attention 레이어의 결과를 모두 concat한 후 또 다른 weight matrix와 연산하여 input과 동일한 쉐잎으로 변형. 
   - 이후 각 행마다(단어마다) fc를 거치고 이때 역시 residual block 기법 사용. 최종 output은 input의 쉐잎, 즉 차원과 동일하다
   - 여기서 중요한 건 output이 input의 차원과 동일하다고 했는데, 그렇기에 encoder를 여러개 이어붙일 수 있게 됨. 그러나, 각각의 encoder들은 weight를 서로 공유하지 않음
   - 실제 논문의 Transformer 모델은 6개의 인코더를 이어붙였음
     

   Decoder Layer

   - Masked Multi head Attention: ??? 이건 모르겠다
   - Multi-head Attention: 현재 디코더 노드의 출력 값을 Query 벡터로 사용하고 인코더에서 Key 와 Value 벡터를 가져와서 연산함.
   - 인코더와 동일하게 residual block
   - 마지막에 Linear 로 피고 Softmax
   - Label Smoothing -> One-hot encoding을 대체한 기법. 너무 학습 데이터에 치중하게 되는 경우 방지하기 위하여 1,0 대신 0부터 1사이의 값으로 부드럽게 펴줌. 0.0001, 0.9, 0.0001 .... 이렇게. (이게 답이긴 한데 저럴 수도 있다)
   - 

- **전체 Process**

  ![img](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

  ![dimension](https://pozalabs.github.io/assets/images/%EC%B0%A8%EC%9B%90.png)

  ![img](/home/jenzzz/Documents/Yoonjin/posts/images/18-21-56.png)

  

  [코드](http://nlp.seas.harvard.edu/2018/04/03/attention.html) | [논문](https://arxiv.org/pdf/1706.03762.pdf)

  

  (+) 이 논문이 발표된 2017년 이후로 지금까지 모델에 많은 발전 및 변화가 있었다. 따라서 논문의 설명과 최신까지 개정되어온 실제 코드 사이의 차이가 조금 있다. 그 디테일을 확인하기 위한 좋은 링크.

  **Details Not Described in the paper** https://tunz.kr/post/4



#### Performance



#### Limitations



#### 이후 고안된 모델들

1. BERT
2. Sparse Transformer
3. Transformer-XL
4. XLNet





#### 4. Sparse Transform

이렇게 Attention is all you need에서 쏘아올린 트랜스포머는 Elmo, Bert 와 같은 모델로 거듭나며 여러 분야의 모델에 대체제로 사용되었고 온갖 state-of-the-art란 art는 다 쓸어담은 게 지난 2017 ~2018년간의 상황이었다. 여전히 attention은 핫하고, 트랜스포머도 핫하다. 이런 시점에서 올해(2019년) 4월 따끈따끈한 논문이 하나 발표되었다. [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf) 은 OpenAI에서 발표한 논문으로 기존 transformer를 컴팩트하게 발전시킨 Sparse Transformer 모델을 소개한다. 

기존 트랜스포머 모델은 시퀀스 길이가 증가하면 증가할수록 메모리 및 계산필요량이 증대한다는 문제가 있었다. 논문은 이를 보완하여 불필요한 계산을 줄이고 압축하거나 recomputation을 통해 메모리 사용량을 줄일 수 있는 방법론을 제시하였고, 추가적으로 residual block과 weight initialization을 재구성하여 결론적으로 매우 딥한 네트워크에 대한 트랜스포머 트레이닝을 상당히 개선시켰다. (Traditional T는 O(n^2 d) 인 반면 Sparse T 는 O(n sqrt(n))) 

그럼 어떻게 했을까? 논문은 다음 세가지를 메인 아이디어로 제시한다.

> 1. A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused).
> 2. An implementation of "strided" and "fixed" attention, as in the Sparse Transformers paper.
> 3. A simple recompute decorator, which can be adapted for usage with attention.



**(1) Attention Learning Pattern**

![1563523560281](/home/jenzzz/Documents/Yoonjin/posts/images/1563523560281.png)

논문 팀은 Transformer의 어텐션 기법이 어떤 패턴으로 학습 되고 있는 지 먼저 연구했다. 실험은 이렇다. 128 레이어의 traditional transformer가 CIFAR-10 데이터셋을 학습하는 과정의 깊이 별로, 다음 pixel을 생성할 때 가장 attention D웨이트가 높은(=가장 주목하고 있는) pixel이 어디인 지 확인하기 위해 해당 pixel들을 밝게 표현한다. 그림 a,b,c,d 는 레이어 깊이 순서대로, 다음 픽셀을 생성해낼 때 attention을 강하게 주는 이전 픽셀들을 밝게(하얗게) 나타내는 그림이다. 레이어 초반부에는 a에서 보다시피 horizontal하게 현재 생성하는 pixel의 주변부만 어텐션 하고 있는 반면 그림 b의 레이어 19와 20에서는 vertically 하게 어텐션 하고 있음을 알 수 있다. 그림 c의 20 ~ 64 사이의 레이어에서는 row와 column 레벨을 벗어나 global data dependent 하게 학습 패턴이 잡혀진 것을 볼 수 있고, 64 레이어 이후부터는 전체 이미지에서 굉장히 적은 양의 픽셀들만을 주목하고 있음을 볼 수 있다(사진에서는 잘 안 보인다). 이 그림을 아주 잘 표현한 움직이는 이미지가 [openai 공식 블로그 포스트](https://openai.com/blog/sparse-transformer/) 에 있다.

오신기하다. 이들은 이렇게 기존 트랜스포머가 무언가를 학습하는 데 있어서 특정 패턴을 보인다는 것, 특히 결국엔 적은 양의 sparse한 데이터에만 attention을 주고 있음에 주목했고, 그렇다면 그냥 처음부터 sparse하게 학습해도 되지 않을까? 라는 아이디어를 내게 되었다.  

![img](/home/jenzzz/Documents/Yoonjin/posts/images/ST_equation1.png)



**(2) Factorized Self-Attention**

![img](/home/jenzzz/Documents/Yoonjin/posts/images/ST_equation234.png)

위와 같은 아이디어를 논문 팀은 'Factorized Self-attention' 이라고 명명했으며 수식은 위와 같다. (개인적으로 이 아이디어의 이름과 시맨틱이 서로 매치되지 않아 혼란스러웠는데, 찾아보니 그냥 루즈하고 hand-wavy(?)하게 이름을 붙인거라 factorized 라는 단어에 너무 신경 쓰지 말라고 한다.) 3번과 4번 식은 transformer의 기존 self-attention 수식과 똑같으나 'S'  라는 친구가 하나 추가되었다. 이 S는 connectiviy pattern, 즉 sparse attention computation method에 의해 쪼개진 input vector indices 집합이다(='set of indices of the input vector'). 즉, S는 총 output vector 크기인 n개의 원소를 가지게 되고, S_i 는 i번째 output vector가 어텐션을 주는 input vector들이 sparse attention method에 의해 쪼개어진 indices 집합이 된다.



(+)

![img](/home/jenzzz/Documents/Yoonjin/posts/images/14-42-28.png)

![img](/home/jenzzz/Documents/Yoonjin/posts/images/14-38-39.png)

논문에서 이 부분, 특히 마지막 문단이 너무 이해가 안 되어서 일단 생각한대로 정리



**(3) Strided & Fixed** 

![1563524350311](/home/jenzzz/Documents/Yoonjin/posts/images/1563524350311.png)

이 논문에서는 sparse attention 연산 메서드로 stride 와 fixed 두개의 approach을 제시한다.  (a)는 현재 i번째 픽셀 이전의 모든 데이터에 대해 연산하는 기존 Transformer, (b), (c)는 Sparse Transformer로서 제안하는 두가지 축소된 attention 기법이다. 

실험 결과 Stride 패턴은 이미지 데이터에, Fixed 패턴은 텍스트 데이터에 적합하다고 함.



*Last Update: 2019/07/30*

*References* 

http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 

http://jalammar.github.io/illustrated-transformer/ 

https://www.youtube.com/watch?v=mxGCEWOxfe8&t=480s 

https://www.youtube.com/watch?v=c2ZiYnFsKEM&t=2094s

