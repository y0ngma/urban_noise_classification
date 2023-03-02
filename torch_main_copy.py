# License: BSD
# Author: Sasank Chilamkurthy
# https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html#finetuning
import copy
import datetime
import json
import os
import random
import time
from collections import Counter
import csv, shutil, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import ResNet50_Weights
from sklearn import metrics
import seaborn as sns
from f1score import final_report
from multiprocessing import freeze_support


def imshow(inp, title=None, save_path: str="."):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.
    filename = f'{datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")}'
    plt.savefig(os.path.join(save_path, f"{filename}.png"), # bbox_inches='tight'
    )


def draw_loss_graph(best_idx, valid_acc, valid_loss, train_acc, train_loss, save_path: str="."):
    ## 결과 그래프 그리기
    print('best model : %d - %1.f / %.1f'%(best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    line_acc_train = ax1.plot(train_acc, 'b-', label='train_acc')
    line_acc_val = ax1.plot(valid_acc, 'r-', label='valid_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')

    ax2 = ax1.twinx()
    line_loss_train =  ax2.plot(train_loss, 'g-', label='train_loss')
    line_loss_val = ax2.plot(valid_loss, 'k-', label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ko')

    ax1.grid()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc', color='k')
    ax2.set_ylabel('loss', color='k')
    ax1.tick_params('y', colors='k')
    ax2.tick_params('y', colors='k')

    lines = line_acc_train+line_acc_val+line_loss_train+line_loss_val
    labels = [ line.get_label() for line in lines ]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    filename = f'{datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")} loss_graph'
    plt.savefig(os.path.join(save_path, f"{filename}.png"), # bbox_inches='tight'
    )


def copy_file_list(test_dataset_size: int, file_dir_list: list, BASE_DIR: str, trg_set_foldername: str):
    """_summary_
    학습시에 8:1:1로 train/validation/test을 나누는데 테스트 데이터셋만 따로 저장하기.
    test에 해당하는 파일경로를 입력받아 테스트셋만 {BASE_DIR}/test/{trg_set_foldername}에 따로 저장해두고
    저장된 모델 성능측정에 사용하도록 한다.

    Args:
        test_dataset_size (int): train_test_split후 테스트 데이터셋의 길이
        file_dir_list (list): 테스트 데이터셋의 실제 파일경로 목록
        BASE_DIR (str): 공통경로인 BASE_DIR하위에 test폴더내에 저장한다
        trg_set_foldername(str): 저장할 폴더명은 구분을 위해 다르게 할 수도 있다

    Examples:
        C:/NIA/dataset/D4a/N-10_220928_D_4_a_37775.jpg 경로의 파일을
        C:/NIA/test/dataset/D4a/N-10_220928_D_4_a_37775.jpg 와 같이 복사한다.
    """
    if not test_dataset_size == len(file_dir_list):
        raise Exception("학습시 테스트 데이터셋으로 나눈 것과 저장할 파일경로 list의 갯수가 다릅니다")

    test_dataset_path = os.path.join(BASE_DIR, "test", trg_set_foldername)
    for file in file_dir_list:
        class_path = os.path.basename(os.path.dirname(file))
        new_path = os.path.join(test_dataset_path, class_path)
        if not os.path.isdir(new_path): os.makedirs(new_path)
        shutil.copy(file, os.path.join(new_path, os.path.basename(file)))
    print('test set 저장 경로', test_dataset_path)

    test_files = glob.glob(os.path.join(test_dataset_path, "**", "*.jpg"), recursive=True)
    if not test_dataset_size == len(test_files):
        raise Exception("학습시 테스트 데이터셋으로 나눈 것과 저장경로내 실제 파일 갯수가 다릅니다")

    csv_dir = os.path.join(test_dataset_path, "test_file_list.csv")
    with open(csv_dir, 'w', encoding='utf-8-sig', newline='') as writefile:
        for row in test_files: csv.writer(writefile).writerow([row])
    # with open(csv_dir, 'r', encoding='utf-8-sig') as f:
    #     for row in csv.reader(f):
    #         print(row)


def train_model(dataloaders, device, save_path, model, criterion, optimizer, scheduler, num_epochs=25, patience=6):
    """patience:몇회 에폭동안 검증 정확도 향상이 없을 시 학습중단할 값. 1이면 향상 없는 에폭에서 바로 중단"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_cnt = -1
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        early_stop_cnt += 1

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                # print(type(dataloaders[phase]))
                # print(labels)
                # print(type(inputs), type(labels))
                # print(len(inputs), len(labels))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"==> best model saved - {best_idx} / {best_acc:.1f}")
                early_stop_cnt = 0

        if early_stop_cnt >= patience:
            print(f"나는.. Ran out of patience... with no improvement for 지난 {early_stop_cnt} epochs 동안,... early stopping 함!")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    time_prefix     = datetime.datetime.today().strftime("%Y%m%d")
    model_save_name = f'{time_prefix}_{round(best_acc)}p@{best_idx}epoch.pt'
    torch.save(model.state_dict(), f"{save_path}/{model_save_name}")
    print('model saved')
    draw_loss_graph(best_idx, valid_acc, valid_loss, train_acc, train_loss, save_path)

    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j],)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test_and_visualize_model(dataloaders, device, class_names, model, criterion, phase = 'val', num_images=4, save_path="."):
    was_training = model.training
    model.eval()
    fig = plt.figure()
    
    running_loss, running_corrects, num_cnt = 0.0, 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size
            # if i == 2: break

        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt       
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc*100))

    # 예시 그림 plot
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 예시 그림 plot
            for j in range(1, num_images+1):
                ax = plt.subplot(num_images//2, 2, j)
                ax.axis('off')
                ax.set_title('%s : %s -> %s'%(
                    'True' if class_names[int(labels[j].cpu().numpy())]==class_names[int(preds[j].cpu().numpy())] else 'False',
                    class_names[int(labels[j].cpu().numpy())], class_names[int(preds[j].cpu().numpy())]))
                imshow(inputs.cpu().data[j], save_path=save_path)
            if i == 0 : break
    model.train(mode=was_training)


def dataset_report(mydatasets: datasets, image_datasets: dict, save_path: str='.'):
    "Make a report of data count per class in each phases"
    print('Counting number of data on each classes...(it takes some time)')
    starttime = datetime.datetime.now()
    train_cnt = dict(Counter([ label for _, label in image_datasets['train'] ]))
    val_cnt   = dict(Counter([ label for _, label in image_datasets['val'] ]))
    test_cnt  = dict(Counter([ label for _, label in image_datasets['test'] ]))
    total_cnt = dict(Counter(mydatasets.targets))
    df        = pd.DataFrame([train_cnt, val_cnt, test_cnt, total_cnt])

    # 실데이터라벨(0,1,2,..) 컬럼명을 클래스명(A1a, A1b,..)으로 바꾸고, 인덱스명도 학습단계명으로 바꿈
    keys = list(mydatasets.class_to_idx.keys())
    vals = list(mydatasets.class_to_idx.values())
    df.rename(columns={key: keys[vals.index(key)] for key in df.columns}, 
                index={0: "train", 1:"val", 2:"test", 3:"total"}, inplace=True)
    # 클래스명이 인덱스로 오고 컬럼명은 "train","validation","test","total"
    df = df.transpose().reset_index(level=0).rename(columns={"index":"class ID"})
    # (클래스별 데이터 총수)/(클래스 총합)*100으로 클래스별 데이터수 비중 계산
    df['ratio to net'] = round(df['total']/df['total'].sum()*100, 2)

    # 클래스명 한글화 필요하여 로드
    reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "소음원별 시나리오 v1.0.xlsx")
    class_df = pd.read_excel(reference_path, usecols=[0,1,2,3])
    df = pd.merge(class_df, df, how='left', on='class ID')

    # 마지막행에 splited subset별 갯수 계산한것 추가
    sum_row = len(df)
    numeric_cols = ['train','val','test','total']
    df.loc[sum_row, [col for col in numeric_cols]] = [df[col].sum() for col in numeric_cols]
    df.loc[sum_row, ['class ID', 'ratio to net']] = ['총계', int(df['ratio to net'].sum())]

    filename = f'{starttime.strftime("%Y%m%d %H-%M-%S")} data_report.xlsx'
    save_dir = os.path.join(save_path, filename)
    df.to_excel(save_dir, index=False)
    print("data report is saved at", save_dir)
    print('lap time:', datetime.datetime.now()-starttime)
    return save_dir


def save_confusion_matrix(testloader: DataLoader, device, model, cls_names, save_path="."):
    was_training = model.training
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    classification_report = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)
    classification_df = pd.DataFrame(classification_report)
    ordered_labels = [ int(val) for val in classification_df.columns if val.isdecimal() ]
    cls_from_test = [ cls_names[val] for val in ordered_labels ]
    print(classification_df.transpose())

    cf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=ordered_labels)
    df_cnt = pd.DataFrame(cf_matrix, index = cls_from_test, columns = cls_from_test)
    filename = f'{datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")} confusion'
    save_dir = os.path.join(save_path, f"{filename}.xlsx")
    df_cnt.to_excel(save_dir)

    return save_dir


def imagefolder_spliter(BASE_DIR, trg_set_foldername, output_path, random_seed):
    """_summary_
        split 안된 데이터셋 경로를 입력하여 imagefolder를 사용하고, 원하는 비율에 따라 split하기
        지속적으로 수집되거나 전처리 방법에 따라 수량이 바뀌는 데이터셋인 경우 매번 split한
        train/val/test set를 직접 저장하지 않고 사용가능하여 편리

    Args:
        BASE_DIR (str): dataset_augmented 데이터셋의 상위 경로
        trg_set_foldername (str): dataset_augmented와 같은 전처리된 폴더명. 
        random_seed (int): int

    Returns:
        _type_: list, Subset
            클래스명 , train set, val set, test set
    """    
    dataset_path = os.path.normpath(os.path.join(BASE_DIR, trg_set_foldername))
    mydatasets = datasets.ImageFolder( dataset_path, transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) )
    image_datasets = {}
    whole_indice = list(range(len(mydatasets)))
    train_idx, tmp_idx      = train_test_split(whole_indice, test_size=0.2, random_state=random_seed)
    image_datasets['train'] = Subset(mydatasets, train_idx)
    val_and_test_indice     = [ idx for idx in whole_indice if idx not in train_idx ]
    val_idx, test_idx       = train_test_split(val_and_test_indice, test_size=0.5, random_state=random_seed)
    image_datasets['val']   = Subset(mydatasets, val_idx)
    image_datasets['test']  = Subset(mydatasets, test_idx)
    dataset_report_dir      = dataset_report(mydatasets, image_datasets, output_path)

    # mydatasets에 대하여 train test split에서 나온 test_idx에 해당하는 것들만 경로 리스트에 담기
    test_file_list = [ mydatasets.__dict__["samples"][idx][0] for idx in test_idx ]
    copy_file_list(len(image_datasets['test']), test_file_list, BASE_DIR, trg_set_foldername)

    return mydatasets.classes, image_datasets['train'], image_datasets['val'], image_datasets['test'], dataset_report_dir


def load_imagefolder(BASE_DIR: str, trg_set_foldername: str):
    """
    dataset_paths = {'train': '/mnt/data/sample_data/train/dataset_augmented_balanced',
                    'val': '/mnt/data/sample_data/val/dataset_augmented_balanced',
                    'test': '/mnt/data/sample_data/test/dataset_augmented_balanced'}
    """
    # 공통주소, 학습단계폴더 및 대상폴더 경로 취합
    dataset_paths = { phase : os.path.normpath(os.path.join(
        BASE_DIR, phase, trg_set_foldername)) for phase in ['train', 'val', 'test'] }
    
    # data_transforms = {
    #     # 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization). 
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     # 검증은 back ground gaussian noise 추가해서 해보자
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    
    image_datasets = {}
    for key, val in dataset_paths.items():
        image_datasets[key] = datasets.ImageFolder(val, transforms.Compose([
                                                        transforms.Resize((224, 224)), 
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    return image_datasets['train'].classes, image_datasets['train'], image_datasets['val'], image_datasets['test']


def my_train(
    BASE_DIR: str,
    trg_set_foldername: str,
    output_path: str,
    EarlyStopPatience: int=3,
    batch_size: int=16,
    number_of_epochs: int=20,
    random_seed: int=555,
    WEIGHTS=None,
    ConvNet_as_fixed_feature_extractor: bool=False,
    Splited_dataset: bool=True
):
    """_summary_

    Args:
        BASE_DIR (str): train/val/test 폴더가 들어있는곳까지의 경로
        trg_set_foldername (str): train/val/test 폴더내의 데이터셋 폴더명. Defaults to dataset_augmented_balanced
        output_path (str): 학습된 모델 및 자동 생성 리포트 저장 경로
        EarlyStopPatience (int, optional): 학습 early stop용 에폭수 . Defaults to 3.
        batch_size (int, optional): _description_. Defaults to 16.
        number_of_epochs (int, optional): _description_. Defaults to 20.
        random_seed (int, optional): _description_. Defaults to 555.
        WEIGHTS (_type_, optional): pretrained된 모델 (ResNet50_Weights.IMAGENET1K_V2)을 적는다. None이면 처음부터 학습. Defaults to None.
        ConvNet_as_fixed_feature_extractor (bool, optional): 마지막 FC층만 학습시 True. Defaults to False.
        Splited_dataset (bool, optional): 데이터셋이 train/val/test 으로 이미 분리된 경우에 기본값 True.
            True 일때: {BASE_DIR}/[train또는 val또는 test]/{trg_set_foldername}/ 의 경로로 데이터셋 경로를 인식하게됨
            False일때: {BASE_DIR}/{trg_set_foldername}/                         의 경로로 데이터셋 경로를 인식하게됨
    """
    freeze_support()
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True
    plt.ion()   # 대화형 모드
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 데이터 로드
    image_datasets = {}
    if Splited_dataset:
        dataset_report_dir = False
        class_names, image_datasets['train'], image_datasets['val'], image_datasets['test'] = load_imagefolder(BASE_DIR, trg_set_foldername)
    else:
        class_names, image_datasets['train'], image_datasets['val'], image_datasets['test'], dataset_report_dir = imagefolder_spliter(BASE_DIR, trg_set_foldername, output_path, random_seed)
    
    dataset_sizes = {key: len(image_datasets[key]) for key in image_datasets.keys()}
    dataloaders   = {key: DataLoader(image_datasets[key], batch_size=batch_size, shuffle=True, num_workers=4) for key in image_datasets.keys()}
    batch_num     = {key: len(dataloaders[key]) for key in image_datasets.keys()}
    print(dataset_sizes)
    print('batch_size: {}, train/val/test: {} / {} / {}'.format(*[batch_size]+[batch_num[key] for key in batch_num.keys()]))

    # 데이터 확인
    plt.ioff()
    show_num_img = len(class_names)
    print(class_names, show_num_img, '개 클래스')
    for key in dataloaders.keys():
        inputs, classes = next(iter(dataloaders[key])) # 학습 데이터의 배치를 얻습니다.
        out = torchvision.utils.make_grid(inputs[:show_num_img], round(np.sqrt(show_num_img))) # 배치로부터 격자 형태의 이미지를 만듭니다.
        imshow(out, title=[class_names[x] for x in classes[:show_num_img]], save_path=output_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    model_ft = models.resnet50(weights=WEIGHTS)

    # 고정된 특징추출기 : backward()중 경사도계산x
    if ConvNet_as_fixed_feature_extractor: 
        for param in model_ft.parameters():
            param.requires_grad = False # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # 모든/마지막 계층의 매개변수들이 최적화되었는지 관찰
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    if ConvNet_as_fixed_feature_extractor:
        optimizer = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

    # 7 에폭마다 0.1씩 학습률 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_ft = train_model(dataloaders, device, output_path, model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=number_of_epochs, patience=EarlyStopPatience)

    ## TEST!
    test_and_visualize_model(dataloaders, device, class_names, model_ft, criterion, phase = 'test', save_path=output_path)

    # 보고서 작성 부분(없어도 학습모델저장 가능)
    cf_df_dir = save_confusion_matrix(dataloaders['test'], device, model_ft, class_names, output_path)
    final_report(cf_df_dir, dataset_report_dir)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    BASE_DIR    = '/mnt/data'
    output_path = os.path.join(BASE_DIR, 'saved_model')
    if not os.path.isdir(output_path): os.makedirs(output_path)

    ### 학습하기
    starttime = datetime.datetime.now()
    print("torch_main.py 모델학습 시작시간 ", starttime)

    my_train(BASE_DIR, 'dataset_augmented', output_path, number_of_epochs=100)
    # my_train(BASE_DIR, 'dataset_augmented_balanced', output_path, number_of_epochs=100, Splited_dataset=False)
    
    print("torch_main.py 모델학습 소요시간 ", datetime.datetime.now()-starttime)