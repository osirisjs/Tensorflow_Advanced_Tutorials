>## **TensorFlow Advanced Tutorials**
        
* ### **Topics** 

    * **니킬부두마의 **딥러닝의 정석**에서 소개하는 내용과 개인적으로 공부한 내용들에 대해 공부하며 작성한 코드들입니다.**  

    * ### **Model With Fixed Length Dataset**
        
        * [***Fully Connected Layer***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_FullyConnectedNeuralNetwork)
            * 기본적인 FullyConnected Neural Network(전방향 신경망) 입니다.

        * [***Convolution Neural Network***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork)

            * 기본적인 Convolution Neural Network(합성곱 신경망) 입니다.
            
            * [ReceptiveField(수용 영역)크기에 대한 계산을 통해 네트워크의 구조를 결정 했습니다.](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/blob/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork/ReceptiveField_inspection/rf.py)

         * **Various Kinds Of Autoencoder**
            * **Feature Extraction Model**
                * [***Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_AutoencoderAndPCA)
                    * 기본적인 Autoencoder 를 PCA 와 비교한 코드입니다.

                * [***Denoising Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_DenoisingAutoencoderAndPCA)
                    * 네트워크의 복원 능력을 강화하기 위해 입력에 노이즈를 추가한 Denoising Autoencoder 를 PCA 와 비교한 코드입니다.

                * [***SparseAutoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_SparseAutoencoderAndPCA)
                    * 소수의 뉴런만 활성화 되는 Sparse Autoencoder 입니다.
            * **Generative Model**

                * [***Basic and Conditional Generative Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_GenerativeAdversarialNetworks)
                    * 무작위로 데이터를 생성해내는 GAN 과 네트워크에 조건을 걸어 원하는 데이터를 생성하는 조건부 GAN 입니다.

                * [***Basic and Conditional Variational Autoencoder***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_VariationalAutoencoder)
                    * Autoencoder를 생성모델로 사용합니다. 짧게 줄여 VAE라고 합니다. 중간층의 평균과 분산에 따라 무작위로 데이터를 생성하는 VAE 와 중간층의 평균과 분산에 target정보를 주어 원하는 데이터를 생성하는 VAE 입니다.
         * **Application**

            * [***LottoNet***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_AutoencoderLottoNet)
                * 로또 당첨의 꿈을 이루고자 전방향 신경망을 사용해 단순히 로또 번호를 예측하는 코드입니다.
                * 네트워크 Graph 구조를 담은 meta 파일을 저장하고 불러오는 코드가 포함되어 있습니다. tensorflow.add_to_collection, tensorflow.get_collection 를 사용합니다.
                * tf.data.Dataset를 사용합니다. 이 API는 자신의 데이터를 학습네트워크에 적합한 형태로 쉽게 처리 할 수 있게 합니다.
            * [***Neural Style***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_NeuralStyle)
                * 내 사진을 예술 작품으로 바꿔주는 유명한 논문인 "A Neural Algorithm of Artistic Style" 의 구현 입니다.
            * [***Word2Vector SkipGram With TSNE***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_Word2Vector_SkipGram_WithTSNE)
                * 아무런 관계가 없는 것처럼 표현(one-hot encoding)된 단어들을 낮은 차원의 벡터로 표현함과 동시에 단어간의 관계를 표현하는 방법입니다. Word2Vector에는 CBOW모델과 Skip-Gram 모델이 있습니다. 여기서는 Skip-Gram 모델을 구현합니다.
            * [***Image To Image Translation With Conditional Adversarial Networks***]()
                * 내가 가지고 있는 도메인의 이미지를 다른 도메인의 이미지로 변환 시켜보자라는 목적을 달성하기 위해 ConditionalGAN 과 UNET을 이용하여 연구를 진행한 논문입니다.
                * 네트워크 구조 및 학습 방법은 논문에서 제시한 내용과 거의 같습니다.(Discriminator 구조인 PatchGAN 의 크기도 70X70 입니다.)
                * edges2shoes 데이터셋을 사용합니다.
                * 효율적으로 데이터를 가져오기위해 TFRecord를 사용합니다.
                * 효율적인 데이터 전처리를 위해 tf.data.Dataset을 사용합니다.

            * [***Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks***]()
                * 진행 예정
            
    * ### **Sequence Model With Variable Length Dataset**
        * ASAP 
    * ### **Reinforcement Learning**
        * ASAP



>## **개발 환경**
* os : ```window 10.1 64bit``` 
* python version(`3.6.4`) : `anaconda3 4.4.10` 
* IDE : `pycharm Community Edition 2018.1.2`
    
>## **코드 실행에 필요한 파이썬 모듈** 
* Tensorflow-1.8.0
* urllib, zipfile, collections
* os, shutil, tqdm
* numpy. matplotlib, scikit-learn, opencv-python

>## **연락처** 
medical18@naver.com