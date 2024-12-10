# Sound-Anomaly-Detection

# 프로젝트 설명

## 1. **프로젝트 소개**
본 프로젝트는 기계 이상음 탐지 모델의 개발을 위해 프리트레인(Pretrain)과 전이학습(Transfer Learning) 기술을 활용한 연구입니다. 기계 이상음은 산업 현장에서 중요한 문제로, 소리만으로 장비의 이상 작동을 감지하면 시스템의 안정성과 가동 시간을 향상시킬 수 있습니다. 

새로운 기계 타입이 등장할 때 기존의 이상음 탐지 모델이 없고, 데이터가 부족하여 새로운 모델을 만드는 것이 어려운 문제입니다. 이를 해결하기 위해 다른 기기들의 데이터를 사용해 학습한 이상음 탐지 모델을 전이학습하여 새로운 기계의 이상음을 탐지하는 방법을 제시할 수 있습니다. 

이 프로젝트에서는 제조업 분야에서 발생하는 다양한 기계의 이상음을 탐지하기 위해, 기존 데이터를 활용한 전이학습이 성능 향상에 도움이 되는지 확인하는 것이 목표입니다.

## 2. **데이터셋 소개**
이 프로젝트에서 사용된 MIMII 데이터셋은 [Zenodo에서 다운로드](https://zenodo.org/record/3384388)할 수 있습니다. 이 데이터셋은 4개의 기계 타입(밸브, 펌프, 팬, 슬라이드 레일)에 대해 정상 작동 및 이상 상태의 오디오 샘플을 제공합니다. 각 기계 유형은 4개의 고유 ID(id_00, id_02, id_04, id_06)를 포함하며, 기계 작동 소음에 다양한 소음 레벨이 추가됩니다. 본 프로젝트에서는 -6 dB 데이터를 사용하였습니다.


## 3. **프로젝트를 위한 데이터셋 구성**
프로젝트는 다음과 같은 파일 및 폴더 구조로 이루어져 있습니다:

1. **ReadMe.md**  
   ***프로젝트 실행 전 읽어야 할 파일***

2. **Sample_data(폴더)**  
   ***프로젝트에서 사용할 데이터***

   - **-6_dB_fan(폴더)**  
     - **fan(폴더)**  
       - **id_00(폴더)**  
         - **normal(폴더)**  
           - 00000000.wav  
           - 00000001.wav  
           - 00000002.wav  
           - ...
         - **abnormal(폴더)**  
           - 00000000.wav  
           - 00000001.wav  
           - 00000002.wav  
           - ...
       - **id_02(폴더)**  
       - **id_04(폴더)**  
       - **id_06(폴더)**

   - **-6_dB_valve(폴더)**  
   - **-6_dB_slider(폴더)**  
   - **-6_dB_pump(폴더)**

3. **Version1(폴더)**  
   ***Version1 실험 관련 폴더***
   - make_pretrain_v1.ipynb
   - make_pretrain_v1.yaml
   - transfer_pretrain_v1.ipynb
   - transfer_pretrain_v1.yaml

4. **Version2(폴더)**  
   ***Version2 실험 관련 폴더***

5. **Version3(폴더)**  
   ***Version3 실험 관련 폴더***

6. **Version4(폴더)**  
   ***Version4 실험 관련 폴더***

7. **install_library.ipynb**  
   ***프로젝트 실행에 필요한 라이브러리 설치 파일***


## 5. **전이학습 Version(v1~v4) 설명**

### [1] Version1(v1)
- 다른 기계 타입으로 pretrain한 모델로 전이학습했을 때 성능 향상을 확인합니다.
  1) **make_pretrain_v1.ipynb** 실행  
     - 이 파일은 pretrain 모델을 생성합니다.  
     - 각 기계 타입별로 정상 소리 파일들을 하나의 데이터셋으로 합쳐 pickle 파일로 저장하고, 이를 이용하여 학습한 모델을 'model_pretrain_v1' 폴더에 저장합니다.
  
  2) **transfer_pretrain_v1.ipynb** 실행  
     - 이 파일은 pretrain 모델을 불러와 전이학습을 진행합니다.  
     - 총 16번의 전이학습을 진행하며, 결과는 'result_transfer_pretrain_v1' 폴더에 저장됩니다.

### [2] Version2(v2)
- 같은 기계 타입의 다른 ID들로 pretrain한 모델을 전이학습했을 때 성능 향상을 확인합니다.
  1) **make_pretrain_v2.ipynb** 실행  
     - 총 16개의 모델을 생성합니다.
  
  2) **transfer_pretrain_v2.ipynb** 실행  
     - 16번의 전이학습을 진행하며, 결과는 'result_transfer_pretrain_v2' 폴더에 저장됩니다.

### [3] Version3(v3)
- 다른 두 기계 타입으로 pretrain한 모델을 전이학습했을 때 성능 향상을 확인합니다.
  1) **make_pretrain_v3.ipynb** 실행  
     - 다양한 기계 타입의 정상 소리 파일을 이용해 모델을 학습합니다.
  
  2) **transfer_pretrain_v3.ipynb** 실행  
     - 전이학습을 통해 16번의 결과를 도출하며, 결과는 'result_transfer_pretrain_v3' 폴더에 저장됩니다.
    
## 6. 수상 경력

- 2023 데이터청년캠퍼스 프로젝트 경진대회 대상
- 2023 데이터 청년인재 양성사업 프로젝트 평가 장려상

## 7. 시연영상

[프로젝트 시연 영상](https://youtu.be/ehZzpnuTtWg?si=Sb7IMBIFaQZq1XWu)
