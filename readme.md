>## **TensorFlow Advanced Tutorials**
        
* ### **Topics** 

    * **니킬부두마의 **딥러닝의 정석**에서 소개하는 내용과 개인적으로 공부한 내용들에 대해 공부하며 작성한 코드들입니다.**  

    * ### **Model With Fixed Length Dataset**
        
        * [***Fully Connected Layer***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_FullyConnectedNeuralNetwork)
            * 기본적인 FullyConnected Neural Network(전방향 신경망) 입니다.

        * [***Convolution Neural Network***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork)

            * 기본적인 Convolution Neural Network(합성곱 신경망) 입니다.

         * **Various Kinds Of Autoencoder**
            * **Feature Extraction Model**
                * [***Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_AutoencoderAndPCA)
                    * 기본적인 Autoencoder를 PCA와 비교한 코드입니다.

                * [***Denoising Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_DenoisingAutoencoderAndPCA)
                    * 네트워크의 복원 능력을 강화하기 위해 입력에 노이즈를 추가한 Denoising Autoencoder를 PCA와 비교한 코드입니다.

                * [***SparseAutoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_SparseAutoencoderAndPCA)
                    * 소수의 뉴런만 활성화 되게 하는 Sparse Autoencoder 입니다.
            * **Generative Model**

                * [***Basic and Conditional Generative Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_GenerativeAdversarialNetworks)
                    * 무작위로 데이터를 생성해내는 GAN과 제한을 두어 원하는 데이터를 생성하는 조건부 GAN 입니다.

                * [***Basic and Conditional Variational Autoencoder***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_VariationalAutoencoder)
                    * Autoencoder를 생성모델로 사용합니다. 줄여서 VAE라고 불립니다. 중간층의 평균과 분산에 따라 무작위로 데이터를 생성하는 VAE와 중간층의 평균과 분산에 target정보를 주어 원하는 데이터를 생성하는 VAE 입니다. - 수정중
         * **Application**

            * [***LottoNet***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_AutoencoderLottoNet)
                * 로또 당첨의 꿈을 이루고자 전방향 신경망을 사용해 단순히 로또 번호를 예측하는 코드입니다.
            * [***Neural Style***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_NeuralStyle)
                * 내 사진을 예술작품으로 바꿔주는 유명한 논문인 "A Neural Algorithm of Artistic Style" 의 구현 입니다.
            * [***Word2Vector SkipGram With TSNE***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_Word2Vector_SkipGram_WithTSNE)
                * 아무런 관계가 없는 것처럼 표현(one-hot encoding)된 단어들을 낮은 차원의 벡터로 표현함과 동시에 단어간의 관계를 표현하는 방법입니다. Word2Vector에는 CBOW모델과 Skip-Gram 모델이 있습니다. 여기서는 Skip-Gram 모델을 구현합니다.
            * [***Image To Image Translation With Conditional Adversarial Networks***]()
                * 진행중
            
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