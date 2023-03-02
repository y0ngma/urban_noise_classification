import glob
import json
import os
import random
import re
import shutil
import numpy as np
import pandas as pd


def Split_data_in_structured_dir(
        GP_dir,
        dist_from_GP=0,
        # trg_ext='.jpg',
        ext_list: list=['.jpg', '.wav'],
        GP_meta: str=None,
        SEED: int=555,
        ratio: dict={"train":0.8, "val":0.1, "test":0.1}):
    """
        Summary:
            대/중/소분류 등이 포함되어 분류된 데이터셋의 최상위 경로에
            train/val/test폴더로 나눠 담기

        GP_dir:
            The parent directory of the largest[highest] categories, so called GrandParent dir.
            In following example of Car dataset,
            C:/dataset/D-segment large cars/KIA/Optima/0001.jpg
            C:/dataset/D-segment large cars/BMW/3-series/0001.jpg
            C:/dataset/D-segment large cars/TOYOTA/camry/0001.jpg
            ...
            the folder structure is so called {segment}/{manufacturer}/{model_name}
            So the largest[highest] category is {segment}.
            In this case, {C:/dataset} is GP_dir.
            Then it makes train, val and test folders and splits the dataset into
            C:/{train, val or test}/{dataset}/{segment}/{manufacturer}/{model_name}.
        
        dist_from_GP:
            The number of depths to the targeted folder directory to split from GP_dir directory level.
            In the above example, if you want to split each {model_name} folders, then it is '3' number of
            depths to reach from dataset folder level. 
            If you set it '0', it mean the targeted folder is parent dir of where each trg_ext files are.

        trg_ext:
            It is to set the filename extension of data, for instance '.jpg', you want to split.
            If other files, for instance '.json', are in the same folder with same filename will be
            paired and copied together to targeted new path.
        
        GP_meta:
            If you want to make a seperated metadata folder like this.
            C:/{train, val or test}/{dataset_JSON}/{segment}/{manufacturer}/{model_name}.
    """
    # Get ext of list to copy all file with same name, e.g., 0001.jpg, 0001.json
    all_files = glob.glob(f"{GP_dir}/**/*{ext_list[0]}", recursive=True)

    # Get folder names of same folder depth level where all of its content to be splited.
    if dist_from_GP==0:
        class_folders = list(set([ os.path.basename(os.path.dirname(file)) for file in all_files ]))
    elif type(dist_from_GP)==int:
        class_folders = list( set([file.split(GP_dir)[-1].split("/")[dist_from_GP] for file in all_files]) )

    # Get a dictionary of file dirs of each class_folder as a value and class_folder as a key
    class_files_dict = {folder:[ file for a_file_list in [glob.glob(f"{GP_dir}/**/{folder}/**/*[{ext}]", recursive=True) for ext in ext_list] for file in a_file_list 
                                ] for folder in class_folders}
    # class_files_dict = {}
    # for folder in class_folders:
    #     file_lists = [ glob.glob(f"{GP_dir}/**/{folder}/**/*[{ext}]", recursive=True) for ext in ext_list ]
    #     files_of_all_ext = [file for a_file_list in file_lists for file in a_file_list ]
    #     class_files_dict[folder] = files_of_all_ext

    # Access dirs in each class and split them into train/val/test after shuffle
    random.seed(SEED)
    for folder, class_files in class_files_dict.items():
        files_no_ext = list( set([os.path.splitext(file_ext)[0] for file_ext in class_files]) )
        random.shuffle(files_no_ext)

        train_dirs = files_no_ext[0:round(len(files_no_ext)*ratio[list(ratio.keys())[0]])]
        rest_dirs  = files_no_ext[round(len(files_no_ext)*ratio[list(ratio.keys())[0]]):]
        val_dirs   = rest_dirs[0:round(len(files_no_ext)*ratio[list(ratio.keys())[1]])]
        test_dirs  = rest_dirs[round(len(files_no_ext)*ratio[list(ratio.keys())[1]]):]

        phase_dict = {list(ratio.keys())[0]:train_dirs, 
                      list(ratio.keys())[1]:val_dirs, 
                      list(ratio.keys())[2]:test_dirs}

        net_cnt = 0 # Number of all files in each class folder to copy which having same filename but different extension
        if GP_meta: meta_cnt = 0
        for phase, phase_dirs in phase_dict.items():
            # Put phase in between, e.g., C:/{train, val or test}/{dataset}
            phase_grand_dir = os.path.join(os.path.dirname(GP_dir), phase, os.path.basename(GP_dir))
            if GP_meta: meta_phase_grand_dir = os.path.join(os.path.dirname(GP_meta), phase, os.path.basename(GP_meta))
            for phase_dir in phase_dirs:
                # Set where to copy files
                # <-   /mnt/data/sample_data/1.원천데이터 / A.층간소음/3.생활소음/d.통돌이세탁기소리 / N-10_220929_A_3_d_11604.jpg     
                # /mnt/data/sample_data/test/1.원천데이터 / A.층간소음/3.생활소음/d.통돌이세탁기소리 / N-10_220929_A_3_d_11604.jpg
                data_filename      = os.path.basename(phase_dir) # 0001.jpg
                data_dirname       = os.path.dirname(phase_dir)
                from_GP_to_dirname = data_dirname.split(GP_dir)[-1] # {segment}/{manufacturer}/{model_name}

                data_trg_dirname = f"{phase_grand_dir}{from_GP_to_dirname}"
                if not os.path.isdir(data_trg_dirname): os.makedirs(data_trg_dirname)
                # Copy files for every filename extension in the class folder 
                for ext in ext_list:
                    data_src_dir = os.path.join(data_dirname, f"{data_filename}{ext}")
                    data_trg_dir = os.path.join(data_trg_dirname, f"{data_filename}{ext}")
                    shutil.copy(data_src_dir, data_trg_dir)
                    net_cnt += 1

                if GP_meta:
                    # <-   /mnt/data/sample_data/2.라벨링데이터 / A.층간소음/3.생활소음/d.통돌이세탁기소리/label / N-10_220929_A_3_d_11604.json 
                    # /mnt/data/sample_data/test/2.라벨링데이터 / A.층간소음/3.생활소음/d.통돌이세탁기소리/label / N-10_220929_A_3_d_11604.json
                    meta_src_dirname = os.path.join(f"{GP_meta}{from_GP_to_dirname}", "label")
                    meta_trg_dirname = os.path.join(f"{meta_phase_grand_dir}{from_GP_to_dirname}", "label")
                    meta_src_dir     = os.path.join(meta_src_dirname, f"{data_filename}.json")
                    meta_trg_dir     = os.path.join(meta_trg_dirname, f"{data_filename}.json")
                    if not os.path.isdir(meta_trg_dirname): os.makedirs(meta_trg_dirname)
                    shutil.copy(meta_src_dir, meta_trg_dir)
                    meta_cnt +=1

        if GP_meta:
            if len(files_no_ext)!=meta_cnt: print(f'WARNING!!! files for {GP_meta:.<32}{folder: <19}:{len(files_no_ext)} is NOT SAME with total moved meta file count:{meta_cnt}')
            # else: print(f'num of files for {GP_meta:.<32}{folder: <19}:{len(files_no_ext)} is SAME with total moved meta file count:{meta_cnt}')
        if len(class_files)!=net_cnt: print(f"WARNING!!! files for {GP_dir:.<33}{folder: <19}:{len(class_files)} is NOT SAME with total moved files count in class folders:{net_cnt}")
        # else: print(f"num of files for {GP_dir:.<33}{folder: <19}:{len(class_files)} is SAME with total moved files count in class folders:{net_cnt}")


def img_to_dataset(sample_path, dest_path, json_too=True, wav_only=False):
    """_summary_
    목적 : 관리하기 용이하도록 폴더구조를 바꾸어 데이터를 분류한다.
        - 사진파일이 담긴 폴더 구조와는 무관하게 파일명에서 classID(D4a 등)을 읽어와서 
        한글이 없도록 classID의 폴더를 만들어 그 안으로 jpg, json 확장자를 가진 파일을 분류.
    Args:
        sample_path (str): 이러한 파일들이 있는 수집완료된 최상위 폴더
        dest_path (str): 분류할 폴더들을 생성할 위치. 이후 이 경로를 모델학습시 데이터셋경로에 적어주어야함.
        json_too (bool, optional): json 파일도 이미지파일과 동일한 경로로 복사시 True.
        wav_only (bool, optional): 사진 대신 오디오파일을 옮긴다. Defaults는 이미지분류에 필요한 사진만 옮긴다.
    Examples:
        다음 경로의 파일을
        C:/1.원천데이터/D.교통소음/4.기타/a.심야에울리는횡단보도신호기소리/N-10_220928_D_4_a_37775.jpg
        C:/2.라벨링데이터/D.교통소음/4.기타/a.심야에울리는횡단보도신호기소리/label/N-10_220928_D_4_a_37775.json
        와 같이 옮긴다
        C:/dataset/D4a/N-10_220928_D_4_a_37775.jpg
        C:/dataset/D4a/N-10_220928_D_4_a_37775.json
    """
    if not wav_only:
        file_list = glob.glob(os.path.join(sample_path,"**", "*.png"), recursive=True)
        file_list += glob.glob(os.path.join(sample_path,"**", "*.jpg"), recursive=True)
    else:
        file_list = glob.glob(os.path.join(sample_path,"**", "*.wav"), recursive=True)
    if len(file_list): print(f"옮길파일수:{len(file_list)}, 파일예시:{file_list[-1]}")
    else: print('옮길 파일이 없습니다')
    cnt = 0
    for file in file_list:
        # 이미지파일명에서 클래스명 조회 N-10_221026_D_4_a_38577.jpg -> D4a
        filename  = os.path.basename(os.path.splitext(file)[0])
        classname = "".join(filename.split("_")[2:5])
        class_path= os.path.join(dest_path, classname)
        regex     = re.compile(r'(\D{1})(\d{1}\D{1})') # (숫자아닌거{0번이상,1이하}번반복)은 그룹1
        matched   = regex.match(classname)
        if matched:
            # print('대분류: ', matched.group(1), '중,소분류:', matched.group(2))
            if not os.path.isdir(class_path):
                os.makedirs(class_path)
            shutil.copy( file, os.path.join(class_path, os.path.basename(file)) )

            if json_too:
                filepath = os.path.dirname(file)
                metapath = os.path.normpath(filepath).split(os.sep)
                metapath[metapath.index("1.원천데이터")] = "2.라벨링데이터"
                metafile = os.path.join("/".join(metapath)+"/label", f"{filename}.json")
                if os.path.isfile(metafile):
                    shutil.copy( metafile, os.path.join(class_path, f"{filename}.json") )
                    cnt +=1
                else: print("json파일 없음:", metafile)
            # if cnt == 10: break
        else:
            print('No match', file)
    if len(file_list)!=cnt: print('Num of copeid files (jpg/json) in', class_path, 'is NOT same', len(file_list),'/' ,cnt)


def dataset_balancer(dataset_src, SEED: int, set_cnt=-1):
    """_summary_
    클래스별 데이터갯수가 상이한 데이터셋(수집속도가 다름, 데이터증강 이후 불균형 등의 이유로)에 
    대하여 갯수가 많은 클래스 일부만 선별하여 데이터정합성을 맞춘 데이터셋 폴더 따로 생성

    Args:
        dataset_src (str): dataset_src 폴더 내에는 다음과 같이 파일이 들어있다
            ㄴ Dog(120장)
                ㄴ dog_pic1.jpg
                ㄴ dog_pic2.jpg
                ...
            ㄴ Cat(600장)
                ㄴ cat_pic1.jpg
                ...
            ㄴ Eagle(650장)
            ...
        SEED: random seed에 사용
        set_cnt (int): 정합성을 맞출 갯수(예:600)를 입력한다. 위의 예시에선 Dog데이터수가 너무 적어
        제외하고 Cat의 수량으로 Eagle의 수량도 맞춘다. 만약 클래스별 데이터 수량중에서 가장
        작은수로 모든 클래스별 데이터수를 맞추려면 -1을 입력한다.
    """
    # 클래스별 데이터수를 파악하기 위해 읽어서 데이터프레임으로 생성
    dataset_pics = glob.glob(os.path.join(dataset_src, "**", "*.jpg"), recursive=True)
    classname_list = [ os.path.basename(os.path.dirname(pic)) for pic in dataset_pics ]
    df_data = list(zip(classname_list, dataset_pics))
    df = pd.DataFrame(df_data, columns=['class', 'file'])

    tmp = df.reset_index(drop=True).groupby('class')['file'].size().sort_values(ascending=False)
    cnt_per_class_df = pd.DataFrame(tmp.values, index=tmp.index, columns = ['cnt'])
    
    # 데이터 정합성(data augmentation으로 늘어난 수 포함)을 위해 설정된 갯수 이상인 클래스명들을 리스트로 반환
    trg_class_list = list(cnt_per_class_df[cnt_per_class_df['cnt']>=set_cnt].index)

    if set_cnt != -1:
        # 정규분포를 활용하여 백분위에 따른 추천 set_cnt 출력
        print('클래스별 갯수가 나오는 데이터프레임 확인')
        print(cnt_per_class_df.sort_values(['cnt'], ascending=False))
        for quantile in [0,3,5]:
            mythreshold = np.quantile(cnt_per_class_df['cnt'].tolist(), quantile/100)
            up_thres_df = cnt_per_class_df[mythreshold<=cnt_per_class_df['cnt']].sort_values(['cnt'])
            recommend_cnt = up_thres_df.head(1)['cnt'].values[0]
            print(f'추천하는 set_cnt는 백분위수 {quantile}% 이상일 때, {recommend_cnt}이고 가용클래스수는 {up_thres_df.shape[0]}개')
        
        print(f"설정된 갯수:{set_cnt} 이상의 데이터수를 가지는 클래스는 총{len(trg_class_list)}/{len(cnt_per_class_df)}개 입니다")

    elif set_cnt == -1:
        set_cnt = np.min(cnt_per_class_df['cnt'].values.tolist())
        trg_class_list = list(cnt_per_class_df[cnt_per_class_df['cnt']>=set_cnt].index)

    # 해당 클래스 파일복사를 위해 반복문으로 접근하고 설정갯수만큼을 섞은 후 복사
    random.seed(SEED)
    save_path = f'{dataset_src}_balanced'
    for cls in trg_class_list:
        src_path = os.path.join(dataset_src, cls)
        trg_path = os.path.join(save_path, cls)
        if not os.path.isdir(trg_path): os.makedirs(trg_path)

        src_list = glob.glob(os.path.join(src_path, "*.jpg"))
        random.shuffle(src_list)
        for src in src_list[-set_cnt:]:
            # filename = os.path.basename(os.path.splitext(src)[0])
            # metafile = os.path.join(os.path.dirname(src), f"{filename}.json")
            # shutil.copy( metafile, os.path.join(trg_path, f"{filename}.json") )
            shutil.copy( src, os.path.join(trg_path, os.path.basename(src)) )
        # print(trg_path, '으로 이동 완료')

    print(f"Completed balancing for {set_cnt} in = ", save_path)
    return save_path


if __name__=="__main__":
    # 새로운 데이터 다운로드시 클래스별 폴더로 단순 분류하기
    BASE_PATH   = "/mnt/data"

    ### 통합본 데이터셋 train test split 한후 전처리 하기
    # train test split 하기
    Split_data_in_structured_dir( GP_dir=os.path.join(BASE_PATH, "1.원천데이터"), 
                                dist_from_GP=0,
                                ext_list=['.jpg', '.wav'],
                                GP_meta=os.path.join(BASE_PATH, "2.라벨링데이터") )