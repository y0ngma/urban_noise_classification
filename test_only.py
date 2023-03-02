import datetime
import os, random
import matplotlib.pyplot as plt
from torch_main_copy import imshow, save_confusion_matrix
from f1score import final_report
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from operator import attrgetter


def test_model(dataloaders, device, class_names, model, criterion, phase = 'val', num_images=4, save_path="."):
    was_training = model.training
    model.eval()
    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    test_size  = len(dataloaders[phase])
    with torch.no_grad():
    # fig = plt.figure()
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs   = inputs.to(device)
            labels   = labels.to(device)
            outputs  = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss     = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

            if i <= num_images:
                print(f"{i}/{test_size} 번째============================================================")
                print("예측:", preds)
                print("정답:", labels.data)
                # 예시 그림 plot
                for j in range(1, num_images+1):
                    ax = plt.subplot(num_images//2, 2, j)
                    ax.axis('off')
                    ax.set_title('%s : %s -> %s'%(
                        'True' if class_names[int(labels[j].cpu().numpy())]==class_names[int(preds[j].cpu().numpy())] else 'False',
                        class_names[int(labels[j].cpu().numpy())], class_names[int(preds[j].cpu().numpy())]))
                    imshow(inputs.cpu().data[j], save_path=save_path)
                    # plt.show()
                    # plt.pause(2)
                    plt.close()
                if i == num_images: print("이하 출력 생략============================================================")

        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt       
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc*100))
    model.train(mode=was_training)


def test_main(
    saved_model_path: str,
    testset_path: str,
    output_path: str,
    batch_size: int=16,
    random_seed: int=555,
):
    freeze_support()
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True

    mydatasets = datasets.ImageFolder( testset_path, transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) )
    class_names = mydatasets.classes
    dataloaders = {'test':DataLoader(mydatasets, batch_size=batch_size, shuffle=True, num_workers=4)}
    show_num_img = len(class_names)
    print(class_names, show_num_img, '개 클래스')

    # 학습된 모델 불러오기
    model_ft = models.resnet50(num_classes=len(class_names))
    # torch.save : pickle 모듈을 이용하여 객체를 직렬화하여 디스크에 저장
    # torch.load : state_dict을 이용하여, 모델 객체 내의 매개변수 값을 초기화 합니다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ckpt = torch.load(saved_model_path, map_location=device)
    # model_ft.load_state_dict(ckpt, strict=False)
    model_ft.load_state_dict(ckpt)
    print(model_ft.fc)
    model_ft = model_ft.to(device)

    # # 모델의 state_dict 출력
    # print("Model's state_dict:")
    # for param_tensor in model_ft.state_dict():
    #     print(param_tensor, "\t", model_ft.state_dict()[param_tensor].size())

    ## TEST!
    criterion = nn.CrossEntropyLoss()
    # 시험수행 시작전/후 시간측정하기
    starttime = datetime.datetime.now()
    print("시험수행 시작시간 ", starttime)
    test_model(dataloaders, device, class_names, model_ft, criterion, phase = 'test', save_path=output_path)

    cf_df_dir = save_confusion_matrix(dataloaders['test'], device, model_ft, class_names, output_path)
    final_report(cf_df_dir)
    print("시험수행 소요시간 ", datetime.datetime.now()-starttime)

def best_of_latest(weight_dirname: str) -> str:
    """_summary_
        경로내의 최근날짜 모델중 정확도가 가장 높은 모델명의 풀경로를 반환, 

    Args:
        weight_dirname (str): 모델이 저장된 폴더의 경로

    Returns:
        str: /saved_model/20230228_85p@60epoch.pt

    예시:infos = ["20230228_77p@40epoch.pt",
                "20230131_99p@999epoch.pt",
                "20230228_85p@60epoch.pt"] 일때 20230228_85p@60epoch.pt을 선택
    """
    class NameInfo:
        def __init__(self, name):
            self.name = name
            self.date = name.split('_')[0]
            self.acc  = name.split('_')[1].split('p@')[0]
            self.epo  = name.split('p@')[1].split('epoch')[0]
        def __repr__(self):
            return repr((self.name, self.date, self.acc, self.epo))
        def __getitem__(self, index):
            return (self.name, self.date, self.acc, self.epo)[index]

    infos = [NameInfo(name) for name in os.listdir(weight_dirname) if 'epoch.pt' in name]
    filename = sorted(infos, key=attrgetter('date', 'acc'), reverse=True)[0][0]

    return os.path.join(weight_dirname, filename)

if __name__ == "__main__":
    BASE_DIR = '/mnt/data'
    testset_path= os.path.join(BASE_DIR, 'test', 'dataset_augmented')
    output_path = os.path.join(BASE_DIR, 'saved_model')
    # weight_path = os.path.join(output_path, "20230104_97p@14epoch.pt")
    weight_path = best_of_latest(output_path)
    print('모델경로:', weight_path)

    # 학습시 사용한 모델의 클래스 수와 테스트할 모델의 클래스수가 같아야 합니다.(전체데이터와 샘플의 클래스수가 다름)
    test_main(weight_path, testset_path, output_path, random_seed=555,)