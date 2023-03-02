# import librosa
# import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os, glob, zipfile, shutil, json
import math, random


def to_melsp(src_path: str, save_path: str)->str:
    """mel-spectrogram으로 변환하기"""
    y, sr = librosa.load(src_path)
    S    = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(16,6))
    librosa.display.specshow(S_DB, sr=sr,hop_length=512, x_axis='time',y_axis='log')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def img_resizer(src_img: str, save_path: str, convert_size: tuple=(224,224)):
    img=Image.open(src_img)
    img=img.convert('RGB')
    img_resize = img.resize(convert_size)
    img_resize.save(save_path)


def kor_unzip(src_zip: str, dest_path: str):
    """src_zip을 해제하여 dest_path에 압축파일명으로 폴더를 생성하고 그 안에 해제한다
    압축파일명에 한글이 있어 cp437로 인코딩 -> euc-kr로 디코딩후 해제"""
    print('해제할 압축파일명:', src_zip)
    dest_folder, _ = os.path.splitext(src_zip)
    dest_folder = os.path.basename(dest_folder)
    dest_path = os.path.join(dest_path, dest_folder)
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
        
    with zipfile.ZipFile(src_zip, "r") as unzipper:
        zipinfo = unzipper.infolist()
        for member in zipinfo:
            member.filename = member.filename.encode("cp437").decode("euc-kr")
            unzipper.extract(member, path=dest_path)
    print('압축해제한 경로명:', dest_path)


def get_size(start_path='.'):
    # walks all sub-directories, summing file sizes
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            # skip if it is symbolic link
            if not os.path.islink(file): total_size += os.path.getsize(file)
    return total_size


def Human_readable_size(size_bytes):
   # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
#    print(size_bytes, '->', s)
   return "%s %s" % (s, size_name[i])


def zip_with_filecount_suffix( _src_path, dest_file, exclude_list=[], add_suffix=True):
    "폴더내 모든 파일을 폴더 구조 유지한 채 압축명 dest_file(~.zip)으로 압축하기"
    cnt = 0
    print(f"압축중... {_src_path:ㅤ<28}: {Human_readable_size(get_size(_src_path))}")
    with zipfile.ZipFile( dest_file, 'w', zipfile.ZIP_DEFLATED ) as zf:
        _rootpath = _src_path
        for ( _dirpath, dirnames, filenames ) in os.walk( _src_path ):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext in exclude_list: continue
                _target_path = os.path.join( _dirpath, filename )
                _rel_path = os.path.relpath( _target_path, _rootpath )
                zf.write( _target_path, _rel_path )
                cnt += 1
    if add_suffix:
        new_file_path = dest_file.split('.')[0]+'_{}EA.zip'.format(cnt)
        try:
            os.rename( dest_file, new_file_path )
            print( '압축이름 {} 압축완료!'.format(new_file_path) )
        except FileExistsError as e:
            print('압축은 완료. 이름바꾸기실패 ',e)
            pass


def show_sizes(BASE_DIR="/mnt/data/NIA", to_zip=False):
    "to_zip=True이면 압축시작"
    rule1 = {"train":"T", "val":"V", "test":"Te"}
    rule2 = {"1.원천데이터":"S", "2.라벨링데이터":"L"}
    for phase, abbr in rule1.items():
        for kind, SL in rule2.items():
            src_path = os.path.join(BASE_DIR, phase, kind)
            dst_path = os.path.join(BASE_DIR, abbr+SL)
            # 용량 확인(가이드라인:압축파일당100기가 이하) 
            print(f"{src_path:ㅤ<28}: {Human_readable_size(get_size(src_path))}")
            # 압축하기
            if to_zip: zip_with_filecount_suffix(src_path, dst_path+'.zip', add_suffix=False)

    GP_dir = f"{BASE_DIR}/train/1.원천데이터"
    class_dirs = sorted(list( set([os.path.dirname(file) for file in glob.glob(f"{GP_dir}/**/*.jpg", recursive=True)]) ))
    mid_class_dirs = sorted(list( set([os.path.dirname(class_dir) for class_dir in class_dirs]) ))
    big_class_dirs = sorted(list( set([os.path.dirname(class_dir) for class_dir in mid_class_dirs]) ))
    print(len(big_class_dirs), '==================================================================')
    for class_dir in big_class_dirs:
        print(f"{class_dir:ㅤ<58}: {Human_readable_size(get_size(class_dir))}")
    print(len(mid_class_dirs), '==================================================================')
    for class_dir in mid_class_dirs:
        print(f"{class_dir:ㅤ<58}: {Human_readable_size(get_size(class_dir))}")
    print(len(class_dirs), '==================================================================')
    for class_dir in class_dirs:
        print(f"{class_dir:ㅤ<58}: {Human_readable_size(get_size(class_dir))}")


def sampling(
        # base_path="/mnt/data/NIA/1.원천데이터",
        base_path="/mnt/data/NIA",
        # label_path="/mnt/data/NIA/2.라벨링데이터",
        class_folders = ["a.심야에울리는횡단보도신호기소리",
        "a.심야에배송트럭이빠르게주행하는소리",
        "a.심야에이륜차가빠르게주행하는소리",
        "a.심야에자동차가빠르게주행하는소리",
        "a.옥외설치확성기의소음",
        "a.등하원아이들떠드는소리",
        "a.골프연습장의타구음",
        "b.항발기의파일뽑는소리",
        "a.덤프트럭의엔진소리",
        "b.샤워할때물소리",
        "a.가구끄는소리",
        "b.고양이우는소리",
        "b.아이들발걸음소리",
        "b.피아노연주소리",],
        SEED = 555,
        sample_per_class = 10,
    ):
    # 14개의 중분류 별 임의의 소분류 한개당 10개 샘플링
    random.seed(SEED)
    for folder in class_folders:
        imgs         = glob.glob(f"{base_path}/1.원천데이터/**/{folder}/*.jpg", recursive=True)

        img_src_dir  = os.path.dirname(imgs[0])
        dir_in_common= img_src_dir.split(f"{base_path}/1.원천데이터")[-1][1:]
        img_dst_dir  = os.path.join(base_path, 'sample', '1.원천데이터', dir_in_common)

        label_src_dir= os.path.join(base_path, '2.라벨링데이터', dir_in_common, 'label')
        label_dst_dir= os.path.join(base_path, 'sample', '2.라벨링데이터', dir_in_common, 'label')

        if not os.path.isdir(img_dst_dir): os.makedirs(img_dst_dir)
        if not os.path.isdir(label_dst_dir): os.makedirs(label_dst_dir)

        random.shuffle(imgs)
        sampled_names  = [ os.path.basename(os.path.splitext(img)[0]) for img in imgs[:sample_per_class] ]
        for nm in sampled_names:
            shutil.copy(f"{img_src_dir}/{nm}.jpg", f"{img_dst_dir}/{nm}.jpg")
            shutil.copy(f"{img_src_dir}/{nm}.wav", f"{img_dst_dir}/{nm}.wav")
            shutil.copy(f"{label_src_dir}/{nm}.json", f"{label_dst_dir}/{nm}.json")


if __name__ == "__main__":
    BASE_DIR = "/mnt/data/NIA"
    # a_zip = os.path.join("/mnt/data", "dummy 데이터.zip")
    # kor_unzip(a_zip, "/mnt/data")

    # show_sizes(BASE_DIR, to_zip=False)

    sampling(sample_per_class=20)
    print(Human_readable_size(get_size(f"{BASE_DIR}/sample")))
    zip_with_filecount_suffix(f"{BASE_DIR}/sample", f"{BASE_DIR}/sample.zip", add_suffix=False)