import glob
import json
import os
import numpy as np
import csv
from PIL import Image
from skimage import io
from sample_handler import img_to_dataset, dataset_balancer
import datetime

def graph_tailor(img, new_shape=(580, 1716, 3)):
    return img[1:new_shape[0], 1:new_shape[1], :]

def crop_with_time_annot(tailored_numpy_img, json_dir, save_path, filename, affix="+", ext=".jpg", ND=3):
    """_summary_
    15초로 통일된 길이로 녹음된 소음원의 멜스팩트로그램 이미지(15초 x축. 1716픽셀)를 5초길이로 잘라 
    정사각으로 만든다. 모델 인풋 사이즈인 224x224로 변형시 왜곡을 줄이면서 가급적 time annotation이 
    있는 구간이 많이 포함되도록 자르고자 한다. time annotation는 실제 소음을 들으면서 해당 구간에 target 소음이 있다고 
    표시한것이다. 휴지기는 target 소음이 없는 경우를 말하는데 자를때 휴지기보다 이러한 구간이 길게끔 시작~끝 지점을
    설정하여 이미지를 크롭하고 가급적 서로 겹치는 부분이 적도록 크롭하여 학습데이터의 품질을 유지하면서
    가공하고자 조건문을 작성하였다.
    Args:
        tailored_numpy_img (_type_): 스펙트로그램x,y축 정보가 크롭된 15초 길이의 원본이미지(numpy형태). 
        json_dir (_type_): time annot json 파일 경로
        save_path (_type_): 크롭 사진 저장 경로
        filename (_type_): 15초길이 원본이미지 파일명
        affix (str, optional): 원본이미지 파일명 뒤에 붙여지는 접미사. Defaults to "+".
        ext (str, optional): 저장할 확장자 설정. Defaults to ".jpg".
        ND (int, optional): Number of Divide약자이고 가로길이(15초)를 3으로 나누는게 기본값. Defaults to 3.
    """
    """
    tailored_numpy_img : 스펙트로그램x,y축 정보가 크롭된 넘파이 이미지
    ND : Number of Divide약자이고 가로길이(15초)를 3으로 나누는게 기본값
    """
    a_jsonfile_dir = os.path.join(json_dir, f"{filename}.json")
    with open(a_jsonfile_dir, "r", encoding='utf-8') as open_json:
        json_temp = json.load(open_json)
    # PIL_img = Image.open(os.path.join(json_dir, f"{filename}{ext}"))
    PIL_img = Image.fromarray(np.uint8(tailored_numpy_img)).convert('RGB')
    X, Y = PIL_img.size
    first_start = json_temp["annotation"][0]["startTime"]
    last_end = json_temp["annotation"][-1]["endTime"]
    first_end = json_temp["annotation"][0]["endTime"]
    last_start = json_temp["annotation"][-1]["startTime"]
    if first_start > 10:
        img_cropped_0 = PIL_img.crop((X/15*last_end-X/ND,0,X/15*last_end, Y))
        img_cropped_0.save( os.path.join(save_path, f'{filename}{affix}0{ext}') )
    else:
        if last_end - first_start <= 5:
            img_cropped_0 = PIL_img.crop((X/15*first_start,0,X/15*first_start+X/ND, Y))
            img_cropped_0.save( os.path.join(save_path, f'{filename}{affix}0{ext}') )
        elif 5 < last_end - first_start <= 10:
            img_cropped_0 = PIL_img.crop((X/15*first_start,0,X/15*first_start+X/ND, Y))
            img_cropped_1 = PIL_img.crop((X/15*last_end-X/ND,0,X/15*last_end, Y))
            img_cropped_0.save( os.path.join(save_path, f'{filename}{affix}0{ext}') )
            img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
        elif last_end - first_start > 10:
            img_cropped_0 = PIL_img.crop((X/15*first_start,0,X/15*first_start+X/ND, Y))
            img_cropped_2 = PIL_img.crop((X/15*last_end-X/ND,0,X/15*last_end, Y))
            img_cropped_0.save( os.path.join(save_path, f'{filename}{affix}0{ext}') )
            img_cropped_2.save( os.path.join(save_path, f'{filename}{affix}2{ext}') )
            if len(json_temp["annotation"]) == 1:
                img_cropped_1 = PIL_img.crop((((last_end - first_start)/2-2.5)*X/15,0,((last_end - first_start)/2-2.5)*X/15+X/ND, Y))
                img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
            elif len(json_temp["annotation"]) == 2:
                if last_start - first_end < 5:
                    img_cropped_1 = PIL_img.crop((((last_end - first_start)/2-2.5)*X/15,0,((last_end - first_start)/2-2.5)*X/15+X/ND, Y))
                    img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
                else:
                    pass
            elif len(json_temp["annotation"]) >= 3:
                max_blank = 0 # annot사이 공백길이
                annotation_index = 0 # 
                time_span = 0
                for i in range(len(json_temp["annotation"])):
                    if json_temp["annotation"][i]["endTime"] - json_temp["annotation"][i]["startTime"] > time_span:
                        time_span = json_temp["annotation"][i]["endTime"] - json_temp["annotation"][i]["startTime"]
                for i in range(len(json_temp["annotation"])-1):
                    if json_temp["annotation"][i+1]["startTime"] - json_temp["annotation"][i]["endTime"] > max_blank:
                        max_blank = json_temp["annotation"][i+1]["startTime"] - json_temp["annotation"][i]["endTime"]
                        annotation_index = i + 1
                if max_blank > 5:
                    if time_span > 5 :
                        img_cropped_1 = PIL_img.crop((((last_end - first_start)/2-2.5)*X/15,0,((last_end - first_start)/2-2.5)*X/15+X/ND, Y))
                        img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
                    else:
                        if json_temp["annotation"][annotation_index]["startTime"] > 10:
                            # 
                            if json_temp["annotation"][annotation_index-1]["endTime"] > 5:
                                img_cropped_1=PIL_img.crop(((json_temp["annotation"][annotation_index-1]["endTime"])*X/15-X/ND,0,(json_temp["annotation"][annotation_index-1]["endTime"])*X/15, Y))
                                img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
                            # 양극단쪽에 annot이 몰린경우(가운데 공백)
                            else:
                                pass
                        else:
                            img_cropped_1 = PIL_img.crop(((json_temp["annotation"][annotation_index]["startTime"])*X/15,0,(json_temp["annotation"][annotation_index]["startTime"])*X/15+X/ND, Y))
                            img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )
                else:
                    img_cropped_1 = PIL_img.crop((((last_end - first_start)/2-2.5)*X/15,0,((last_end - first_start)/2-2.5)*X/15+X/ND, Y))
                    img_cropped_1.save( os.path.join(save_path, f'{filename}{affix}1{ext}') )

def error_list_writer(save_path, error_list):
    "에러발생 파일명 저장하기"
    error_list = list(set(error_list))
    if error_list:
        with open(f"{save_path}_error.csv", 'a', encoding='utf-8-sig', newline='') as write_file:
            wr = csv.writer(write_file)
            wr.writerow(["filename"])
            for line in error_list: wr.writerow([line])

class MyTransform(object):
    def __init__(self, class_pics, save_path, error_list):
        self.number_of_slicing = 3
        # error_list = list()
        self.pic = class_pics
        self.save_path = save_path
        self.error_list = error_list

    def __call__(self):
        pic_name    = os.path.basename(self.pic)
        pic_dir     = os.path.dirname(self.pic)
        cls_name    = os.path.basename(pic_dir)
        augment_dir = os.path.join(self.save_path, cls_name)
        if not os.path.isdir(augment_dir): 
            print("다음 경로 생성됨", augment_dir)
            os.makedirs(augment_dir)
        filename, ext = os.path.splitext(pic_name)

        img = io.imread(self.pic)
        #### 사진 외곽 자르기
        self.tailored = graph_tailor(img)

        #### time annotation 포함하게 저장
        crop_with_time_annot(self.tailored, pic_dir, augment_dir, filename, "+", ext)



def preprocess(
        original_path: str='/mnt/data/1.원천데이터',
        dataset_path: str='/mnt/data/dataset',
        move_json_too: bool=True,
        move_wav_only: bool=False,
        balance_data: bool=True
        ):
    """_summary_
    original path의 jpg, json파일을 이용하여 전처리 후 따로 저장. 각 단계별 처리마다 
    1-1. img_to_dataset() : dataset/
    1-2. mytrans() : dataset_augment/
    1-3. dataset_balancer() : dataset_augmented_balanced/
    폴더가 생성되고 파일을 복사하여 확인하기 용이하나, 용량유의. 
    [경우2: train test split된 데이터 각각 전처리 하기] 진행시 1-3.은 진행하지 않아서,
    original_path의 audio파일 제외(기본값)하고 단계별 7.6GB 증가. 
    최소 16GB 추가공간필요(전처리 완료 후 dataset폴더는 삭제가능)
    
    Args:
        original_path (str, optional): Defaults to '/mnt/data/1.원천데이터'.
        dataset_path (str, optional): Defaults to '/mnt/data/dataset'.
        move_json_too (bool, optional): Defaults to True.
        move_wav_only (bool, optional): Defaults to False.
        balance_data (bool, optional): Defaults to True.
    """    
    # 1-1. sample_handler 대/중/소 분류하는 구조 없이 영문명 소분류 폴더로 복사
    img_to_dataset(original_path, dataset_path, json_too=move_json_too, wav_only=move_wav_only)
    # 1-2. 데이터 전처리
    augment_path = f"{dataset_path}_augmented"
    error_list = list()
    for cls_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, cls_name)
        class_pics = glob.glob(os.path.join(class_path, "**", "*.jpg"), recursive=True)
        for pic in class_pics:
            mytrans = MyTransform(pic, save_path=augment_path, error_list=error_list)
            mytrans()
    # 1-3. 데이터정합성 -> 특정갯수로 전클래스 갯수 통일시키기(기본값:클래스별 데이터수가 가장 적은 것)
    if balance_data: dataset_balancer(augment_path, SEED=555, set_cnt=-1)

    
if __name__=="__main__":
    BASE_PATH   = "/mnt/data"

    starttime = datetime.datetime.now()
    print("데이터 전처리 및 증강 시작시간 ", starttime)
    
    # ### 경우1: train test split안된 통합본 데이터셋 전처리 하기
    # preprocess(os.path.join(BASE_PATH, "sample", "1.원천데이터"), os.path.join(BASE_PATH, "sample", "dataset"))

    ### 경우2: train test split된 데이터 각각 전처리 하기
    for phase in ["train", "val", "test"]:
        phase_path   = os.path.join(BASE_PATH, phase)
        dataset_path = os.path.join(phase_path, 'dataset')
        preprocess(os.path.join(phase_path, "1.원천데이터"), dataset_path, balance_data=False)

    print("데이터 전처리 및 증강 소요시간 ", datetime.datetime.now()-starttime)