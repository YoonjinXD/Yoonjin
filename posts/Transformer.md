### Transformer

앞서 RNN의 첫등장 -> 문장이 길어질수록 앞의 내용을 잊어버리는 Long-term dependency 문제 -> 그로 인해 고안되었던 attention 기법,  LSTM, 그리고 GRU 라는 성과들에도 불구하고, 이들은 모두 순차적으로 정보를 읽어들이는 RNN 기반 구조였기에 Long-term dependency를 완전히 탈출하지는 못했다. 이러한 한계점을 타개해준 논문이 바로 그 유명한 'Attention is all you need' 이다. 어텐션이 어텐션 받고 있는 만큼 딥러닝에 관심 있다면 반드시 한번쯤 읽어보는 것이 좋다. 이 논문은 제목 그대로, 기존 RNN 기반 모델들을 아예 사용하지 않고 attention 기법만을 적용해도 sequence 모델을 다루는 데 손색이 없고 심지어 더 좋은 성능과 결과를 보인다는 내용이다. 

주요 아이디어는 대략 이렇다. 문장에서 중요한 부분만 기억하며 순차적으로 읽어나가는 것이 아니라, 병렬화(Parallelization)을 통해 한 방에 문장을 위에서 관조하자! 전체 문장에서 주목해야할 부분에 가중치를 높이는 방식이기 때문에 어떠한 정보를 잊어버릴 일이 없다. 즉, Long-term dependency를 완벽히 극복~~(비록 또다른 문제가 발생하긴 했지만)~~해낸 것이다. 이 논문에서 제시한 모델이 바로 이름부터 멋있는**Transformer** 이고 이것은 여러 분야에 응용되며 다양한 분야의 state-of-the-art 자리에 앉게 되었다.
(+) 이 논문 이후 해당 아이디어를 응용한 모델들이 마구 나와서, 사실 원조 Transformer 자체로는 SOTA 자리를 오래 차지하지 못했다고 한다(ex. 춘추전국시대).



#### Overview

![img](/home/jenzzz/Documents/Yoonjin/posts/images/transformer.png)

*[Figure 1] Transformer architecture -ref [paper](https://arxiv.org/pdf/1706.03762.pdf)*





#### Modules

1. Input Embedding
   

2. Positional Encoding
   

3. **Multi-Head Attention**
   Transformer에서 가장 핵심적인 메커니즘으로, 이를 이해하기 위해 먼저 논문에서 제안한 **Self-attention** 알고리즘을 먼저 살펴 보자. 

   * Self Attention

     ![img](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

     *[Figure] Self-attention 에 대한 직관을 잘 보여주는 그림*

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

   이제 Multi-head attention 중에 'attention' 까지는 이해했다. 남은 'Multi-head'는 그래도 비교적 간단하게, 위와 같은 scaled-dot product 연산을 여러개의 쓰레드(=head)로 내려 한 단어에 대해 여러번 attention 정보를 얻어내겠다는 것이다. 예를 들어, 만약 head가 4개라면 4개의 attention 벡터가 생성되고 이 벡터들을 concat하여 최종 attention 벡터를 도출하게 된다. 따라서 각 head의 attention 벡터 차원 곱하기 head 개수는 반드시 모델의 차원이 되어야 한다.

   여기 그 수식 그림

   근데 이런 일을 왜 할까? 어떤 문장의 단어는 2~3개의 단어와 밀접한 연관을 가지고 있을 수 있고, 또는 모호한 애매한 관계를 가지고 있는 단어들 역시 처리할 수 있기 때문이다. 

   

4. Feed Forward
   

5. Encoder and Decoder
   Encoder Layer

   - Positional Encoding 이 Back prop때 vanishing 되는 현상을 방지하기 위해 Residual Block 기법을 사용하여 Positional 정보를 전달.
   - Multi-head Attention 레이어의 결과를 모두 concat한 후 또 다른 weight matrix와 연산하여 input과 동일한 쉐잎으로 변형. 
   - 이후 각 행마다(단어마다) fc를 거치고 이때 역시 residual block 기법 사용. 최종 output은 input의 쉐잎, 즉 차원과 동일하다
   - Normalization 당연잇음
   - 여기서 중요한 건 output이 input의 차원과 동일하다고 했는데, 그렇기에 encoder를 여러개 이어붙일 수 있게 됨. 그러나, 각각의 encoder들은 weight를 서로 공유하지 않음
   - 실제 논문의 Transformer 모델은 6개의 인코더를 이어붙였음
     

   Decoder Layer

   - Masked Multi head Attention: ??? 이건 모르겠다
   - Multi-head Attention: 현재 디코더 노드의 출력 값을 Query 벡터로 사용하고 인코더에서 Key 와 Value 벡터를 가져와서 연산함.
   - 인코더와 동일하게 residual block
   - 마지막에 Linear 로 피고 Softmax
   - Label Smoothing -> One-hot encoding을 대체한 기법. 너무 학습 데이터에 치중하게 되는 경우 방지하기 위하여 1,0 대신 0부터 1사이의 값으로 부드럽게 펴줌. 0.0001, 0.9, 0.0001 .... 이렇게. (이게 답이긴 한데 저럴 수도 있다)



#### Performance



#### Limitations



#### 이후 고안된 모델들

1. BERT
2. Sparse Transformer
3. Transformer-XL
4. XLNet