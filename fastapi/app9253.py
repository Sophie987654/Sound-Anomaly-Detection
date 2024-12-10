from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import logging
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import sys
import numpy
from sklearn import metrics
import wave
import math
from pydub import AudioSegment
from tqdm import tqdm


# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 초기화
app = FastAPI()

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 홈 페이지 렌더링
@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index79.html', {"request": request})


# 사용자의 데이터로 재학습하는 페이지
@app.get('/transfer_page')
async def transfer_page(request: Request):
    return templates.TemplateResponse('second3.html', {"request": request})

# 재학습된 사용자의 모델로 예측하는 페이지
@app.get('/predict_page')
async def predict_page(request: Request):
    return templates.TemplateResponse('third7.html', {"request": request})

# 사전 모델로 예측하는 페이지
@app.get('/pretrained_predict_page')
async def pretrained_predict_page(request: Request):
    return templates.TemplateResponse('first6.html', {"request": request})


# 모델 로드
models = {
    "fan": tf.keras.models.load_model("./model_transfer_pretrain_v1/pretrain_only_pump-6.h5"),
    "valve": tf.keras.models.load_model("./model_transfer_pretrain_v1/pretrain_only_slider-6.h5"),
    "pump": tf.keras.models.load_model("./model_transfer_pretrain_v1/pretrain_('valve', 'slider').h5"),
    "slider": tf.keras.models.load_model("./model_transfer_pretrain_v1/pretrain_('fan', 'valve', 'pump').h5")
}

# 임계값 설정
thresholds = {
    "fan": 8.619176427417994,
    "valve": 9.897634201965031,
    "pump": 9.563051750767473,
    "slider": 9.783154623572726
}



# 사용자 정의 모델 저장소
custom_models = {}

# 함수 정의
# 함수 정의
def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data
        return sr, np.array(multi_channel_data)[channel, :]
    except ValueError as msg:
        logger.warning(f'{msg}')

def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    sr, y = demux_wav(file_name) 
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    return vectorarray


# wav파일 넣어 예측하는 부분 : ver 1
@app.post('/predict')
async def predict(audio: UploadFile = File(...), machineType: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            processed_data = file_to_vector_array(tmp.name)
        
        selected_model = models[machineType]

        prediction = selected_model.predict(processed_data)
        error = np.mean(np.square(processed_data - prediction), axis=1)
        mean_error = numpy.mean(error)
        logger.info(f"Mean squared error for each sample: {error}")
        logger.info(f"mean error: {mean_error}")
        
        if machineType == 'fan':
            if mean_error > thresholds[machineType]:
                result = "정상"
            else:
                result = "비정상"
        
        elif machineType == 'pump':
            if mean_error > thresholds[machineType]:
                result = "정상"
            else:
                result = "비정상"
        
        else:
            if mean_error > thresholds[machineType]:
                result = "비정상"
            else:
                result = "정상"


        return {"result": result, "errorData": {"error": error.tolist(), "time_steps": list(range(len(error)))}}

    except Exception as e:
        logger.exception("Exception occurred!")
        return JSONResponse(status_code=400, content={"error": str(e)})

    
# ver2 시작

transfer_v1_param = {
    "pickle_directory": "./pickle_transfer_pretrain_v1",
    "model_directory": "./model_transfer_pretrain_v1",
    
    "feature": {
        "n_mels": 64,
        "frames": 5,
        "n_fft": 1024,
        "hop_length": 512,
        "power": 2.0
    },

    
    "fit": {
        "compile": {
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metrics": ["accuracy"]
        },
        "epochs": 20,
        "batch_size": 32,
        "shuffle": True,
        "validation_split": 0.2,
        "verbose": 1
    }
}

os.makedirs(transfer_v1_param["pickle_directory"], exist_ok=True)
os.makedirs(transfer_v1_param["model_directory"], exist_ok=True)

def get_pretrained_model_path(machine_type):
    model_paths = {
        "fan": "./model_transfer_pretrain_v1/pretrain_only_pump-6.h5",
        "valve": "./model_transfer_pretrain_v1/pretrain_only_slider-6.h5",
        "pump": "./model_transfer_pretrain_v1/pretrain_('valve', 'slider').h5",
        "slider": "./model_transfer_pretrain_v1/pretrain_('fan', 'valve', 'pump').h5"
    }
    return model_paths.get(machine_type)

# 함수: WAV 파일을 10초 단위로 자르고 리스트로 반환
def split_wav_file(input_wav_path, output_dir, duration=10):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the input WAV file
    y, sr = librosa.load(input_wav_path, sr=None)

    # Calculate the total duration in seconds
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the number of segments
    num_segments = int(total_duration / duration)

    file_list = []

    for i in range(num_segments):
        # Calculate the start and end times for the segment
        start_time = i * duration
        end_time = (i + 1) * duration

        # Ensure the end time does not exceed the total duration
        if end_time > total_duration:
            end_time = total_duration

        # Extract the segment from the audio
        segment = y[int(start_time * sr):int(end_time * sr)]

        # Define the output filename for the segment
        output_filename = os.path.join(output_dir, f"segment_{i}.wav")

        # Save the segment as a WAV file
        sf.write(output_filename, segment, sr)

        file_list.append(output_filename)

    return file_list


# WAV 파일 리스트를 벡터 배열로 변환하는 함수
#소리 파일들의 이름명이 담긴 리스트를 입력하면 그것들을 하나의 데이터셋으로 합치는 함수
def list_to_vector_array(file_list, msg="calc...", n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx], n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)          
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset


# 2단계 : 업로드 된 wav파일을 pretrain 모델에 전이학습시키기
@app.post('/transfer')
async def transfer(audio: UploadFile = File(...), machineType: str = Form(...), modelName: str = Form(...)):
    try:
        # 업로드된 WAV 파일을 임시 파일로 저장하고 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_wav_file = tmp.name
        
        # 잘라진 WAV 파일 리스트를 생성
        output_segment_dir = "output_segments"
        os.makedirs(output_segment_dir, exist_ok=True)
        file_list = split_wav_file(tmp_wav_file, output_segment_dir,duration=10)
        
        # 모델 경로 가져오기
        pretrained_model_path = get_pretrained_model_path(machineType)
        
        # 모델 로드
        model = tf.keras.models.load_model(pretrained_model_path)
        
        # 각 WAV 파일에 대한 벡터 배열 리스트 생성
        train_data = list_to_vector_array(file_list,
                                          msg="generate train_dataset",
                                          n_mels=transfer_v1_param["feature"]["n_mels"],
                                          frames=transfer_v1_param["feature"]["frames"],
                                          n_fft=transfer_v1_param["feature"]["n_fft"],
                                          hop_length=transfer_v1_param["feature"]["hop_length"],
                                          power=transfer_v1_param["feature"]["power"])
        
        
        # 전이학습 수행
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])
        model.fit(train_data,
                  train_data,
                  epochs=transfer_v1_param["fit"]["epochs"],
                  batch_size=transfer_v1_param["fit"]["batch_size"],
                  shuffle=transfer_v1_param["fit"]["shuffle"],
                  validation_split=transfer_v1_param["fit"]["validation_split"],
                  verbose=transfer_v1_param["fit"]["verbose"])
        
        # 새로운 모델 저장
        new_model_path = os.path.join(transfer_v1_param["model_directory"], f"{modelName}.h5")
        model.save(new_model_path)

        # 사용자 정의 모델 저장소에 저장
        custom_models[modelName] = {"model": model, "machineType": machineType}


        return {"result": "전이학습이 완료되었습니다."}

    except Exception as e:
        logger.exception("Exception occurred!")
        return JSONResponse(status_code=400, content={"error": str(e)})
        

# 3단계 : 전이학습된 모델에 wav파일을 업로드하여 이에 대한 예측값을 출력
import random  # random 모듈을 import합니다.

def split_wav_file_random(input_wav_path, output_dir, duration=10):
    # 결과를 저장할 디렉터리를 생성합니다.
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 WAV 파일을 로드합니다.
    y, sr = librosa.load(input_wav_path, sr=None)

    # 오디오의 총 길이(초)를 계산합니다.
    total_duration = librosa.get_duration(y=y, sr=sr)

    # 오디오를 분할할 세그먼트 수를 계산합니다.
    num_segments = int(total_duration / duration)

    # 무작위로 세그먼트를 선택합니다.
    random_segment_index = random.randint(0, num_segments - 1)

    # 선택된 세그먼트의 시작과 끝 시간을 계산합니다.
    start_time = random_segment_index * duration
    end_time = (random_segment_index + 1) * duration

    # 끝 시간이 총 길이를 초과하지 않도록 합니다.
    if end_time > total_duration:
        end_time = total_duration

    # 오디오에서 선택된 세그먼트를 추출합니다.
    segment = y[int(start_time * sr):int(end_time * sr)]

    # 선택된 세그먼트를 저장할 파일 경로를 정의합니다.
    output_filename = os.path.join(output_dir, f"selected_segment.wav")

    # 선택된 세그먼트를 WAV 파일로 저장합니다.
    sf.write(output_filename, segment, sr)

    return output_filename


# 사용자 정의 모델로 예측
@app.post('/predict_custom')
async def predict_custom(audio: UploadFile, modelName: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())

        # 로그 출력
        print(f"Received audio file for model: {modelName}")

        model_data = custom_models.get(modelName)
        if not model_data:
            raise Exception("해당 이름의 모델을 찾을 수 없습니다.")
        selected_model = model_data["model"]
        machineType = model_data["machineType"]
        # 로그 출력
        print(f"Machine Type: {machineType}")

        # 수정된 부분: split_wav_file_random 함수를 사용하여 무작위로 선택된 WAV 파일을 가져옵니다.
        selected_wav_file = split_wav_file_random(tmp.name, "selected_segments")

        # 선택된 WAV 파일을 처리하고 벡터 배열로 변환합니다.
        processed_data = file_to_vector_array(selected_wav_file,
                                              n_mels=transfer_v1_param["feature"]["n_mels"],
                                              frames=transfer_v1_param["feature"]["frames"],
                                              n_fft=transfer_v1_param["feature"]["n_fft"],
                                              hop_length=transfer_v1_param["feature"]["hop_length"],
                                              power=transfer_v1_param["feature"]["power"])

        prediction = selected_model.predict(processed_data)
        error = np.mean(np.square(processed_data - prediction), axis=1)
        mean_error = numpy.mean(error)
        
        # 로그 출력
        print(f"Mean Error: {mean_error}")

        if machineType == 'fan':
            if mean_error > 190:
                result = "정상"
            else:
                result = "비정상"
        
        elif machineType == 'pump':
            if mean_error > thresholds[machineType]:
                result = "정상"
            else:
                result = "비정상"
        
        else:
            if mean_error > thresholds[machineType]:
                result = "비정상"
            else:
                result = "정상"

        # 예측 결과 및 에러 데이터를 반환합니다.
        return {"result": result, "errorData": {"error": error.tolist(), "time_steps": list(range(len(error)))}}

    except Exception as e:
        logger.exception("Exception occurred during prediction!")
        return JSONResponse(status_code=400, content={"error": str(e)})




@app.get('/custom_model_names')
async def get_custom_model_names():
    return {"modelNames": list(custom_models.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)








