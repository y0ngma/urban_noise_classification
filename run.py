import argparse
# reference 주소 https://wikidocs.net/73785
from torch_main_copy import my_train
from custom_preprocessor import preprocess
import os, datetime

def main(BASE_DIR: str, trg_set_foldername: str, output_path: str, num_of_epoch: int, Splited_dataset: bool=True):
    """_summary_
    샘플데이터 압축해제, 전처리~학습
    Args:
        BASE_DIR (str): "/mnt/data/"
        trg_set_foldername (str): 학습에 사용할 전처리된 폴더명(dataset_augmented). 
        output_path (str): 학습 산출물(모델, 학습그래프, sklearn 오차행렬 엑셀 등) 저장 경로
        num_of_epoch (int): 학습 에포크 설정. 
        Splited_dataset (bool): train/val/test나눠져 있지 않은 샘플데이터 등 사용시 False.
    """
    # assert check parameters
    if not os.path.isdir(output_path): os.makedirs(output_path)

    starttime = datetime.datetime.now()
    print("데이터 전처리 및 증강 시작시간 ", starttime)

    ### 경우1: train test split안된 통합본 데이터셋 전처리 하기
    if not Splited_dataset:
        dataset_path = os.path.join(BASE_DIR, 'dataset')
        preprocess(os.path.join(BASE_DIR, "1.원천데이터"), dataset_path, )

    ### 경우2: train test split된 데이터 각각 전처리 하기
    else:
        for phase in ["train", "val", "test"]:
            phase_path   = os.path.join(BASE_DIR, phase)
            dataset_path = os.path.join(phase_path, 'dataset')
            preprocess(os.path.join(phase_path, "1.원천데이터"), dataset_path, balance_data=False)

    print("데이터 전처리 및 증강 소요시간 ", datetime.datetime.now()-starttime)

    ### 학습하기
    starttime = datetime.datetime.now()
    print("torch_main.py 모델학습 시작시간 ", starttime)
    my_train(BASE_DIR, trg_set_foldername, output_path, number_of_epochs=num_of_epoch, Splited_dataset=Splited_dataset)
    print("torch_main.py 모델학습 소요시간 ", datetime.datetime.now()-starttime)

if __name__ == "__main__":
    # # Argument(전달인자) : parameter(매개변수)에 전달되는 값
    # parser = argparse.ArgumentParser()
    # # 추가옵션을 받는 경우 action="store"
    # parser.add_argument("-e", "--epochs", "--epoch", dest="epochs", action="store")
    # parser.add_argument("-b", "--batch_size", dest="batch_size", action="store")
    # # 추가옵션을 받지 않고 단지 옵션의 유/무만 필요한경우 action="store_true"
    # parser.add_argument("-r", "--report", dest="make_data_report", action="store_true")
    # parser.add_argument("-g", "--gpu", dest="use_gpu", action="store_true")
    # parser.add_argument("-p", "--pretrained", dest="use_pretrained", action="store_true")
    # # # existence / non existence
    # # args = parser.parse_args() # python NIA/run.py --epoch 1 -b 16 -r -g 
    # # command line argument 미사용시 인자로 리스트를 넘겨준다
    # args = parser.parse_args(["--epoch", "1", "-b", "16", "-r", "-g"])

    # if args.epochs <= "1": print("epochs should bigger than 1 ")
    # if args.batch_size: print("배치사이즈 : ", args.batch_size)
    # if args.make_data_report: print("데이터셋보고서 생성: ", args.make_data_report)
    # if args.use_gpu: print("GPU 사용: ", args.use_gpu)
    # if args.use_pretrained: print("pretrained 모델 사용: ", args.use_pretrained)
#############################################################
    BASE_DIR     = "/mnt/data"
    output_path  = os.path.join(BASE_DIR, 'saved_model')
    # main(BASE_DIR, 'dataset_augmented', output_path, num_of_epoch=30, Splited_dataset=True)

    sample_dir   = os.path.join(BASE_DIR, "sample")
    main(sample_dir, 'dataset_augmented_balanced', output_path, num_of_epoch=30, Splited_dataset=False)