# https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, sys, glob


def font_path_finder(search_keyword, search_keyword_2nd=None):
    """한글폰트 설치 후 한글폰트명의 영문명을 모를때 폰트검색 후 경로반환하기"""
    target_font = list()
    if sys.platform == 'win32':
        font_list = fm.findSystemFonts()
        font_list.sort()
        for file_path in font_list:
            font_attribute = fm.FontProperties(fname=file_path)
            font_name = font_attribute.get_name()
            if search_keyword in font_name:
                print(f"폰트이름 : {font_name}")
                print(f"폰트속성 : {font_attribute}")
                print(f"폰트경로 : {file_path}") # "C:/Users/USER/AppData/Local/Microsoft/Windows/Fonts/MaruBuri-Regular.ttf"
                target_font.append(file_path)
    return target_font


def summarize_confusion_matrix(original_matrix: pd.DataFrame):
    """
    matrix는 오차행렬 : sklearn.metrics.confusion_matrix(y_true, y_pred)
        # Constants
        C, F, H = "Cat", "Fish", "Hen"
        # True values
        y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H]
        # Predicted values
        y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H]
    """
    matrix = original_matrix.copy()
    TP_plus_TN_plus_FP_plus_FN = int( np.nansum(matrix) )
    cls_list = list(matrix.index)
    print(f"Confusion Matrix shape:{matrix.shape}, TP+TN+FP+FN={TP_plus_TN_plus_FP_plus_FN}")
    for val in cls_list:
        TP = matrix.loc[val, val]
        TN_idx = [ idx for idx in cls_list if not idx == val ]
        TN = int( np.nansum(matrix.loc[TN_idx, TN_idx]) )
        TP_plus_FP = int( np.nansum(list(matrix.loc[:, val].values)) )
        TP_plus_FN = int( np.nansum(list(matrix.loc[val, :].values)) )
        Recall = TP/TP_plus_FN
        Precision = TP/TP_plus_FP
        Accuracy = (TP+TN)/TP_plus_TN_plus_FP_plus_FN

        # tp+fn, recall 컬럼 추가
        matrix.loc[val, "total"] = TP_plus_FN
        matrix.loc[val, "recall"] = round(Recall, 3)
        # tp+fp, precision 행 추가
        matrix.loc["total", val] = TP_plus_FP
        matrix.loc["precision", val] = round(Precision, 3)
        # precision, f1-score 컬럼 추가
        matrix.loc[val, "precision"] = round(Precision, 3)
        matrix.loc[val, "f1-score"] = round(2*Precision*Recall/(Precision+Recall), 3)
        matrix.loc[val, "accuracy"] = round(Accuracy, 3)
    
    # 모든 cls_list의 recall,precision,f1-score,accuracy를 구한 후 합계 및 평균 구하기
    matrix.loc["total"] = matrix.loc[cls_list].sum(axis=0)
    matrix.loc["mean"] = round(matrix.loc[cls_list].mean(axis=0), 3)
    summary = matrix.loc[cls_list+["mean"], ["recall","precision","f1-score","accuracy",]]
    summary.index.name = "class ID"
    return matrix, summary


def info_df_merge(target_df: pd.DataFrame, reference_df_dir=False):
    """
    리포트내 클래스명 한글화 필요하여 참조하여 merge할 df 로드
    reference_df_dir : 왼쪽에 class ID로 merge 할 엑셀파일경로
    target_df : 우측에 인덱스로 merge할 데이터프레임 
    """
    if not reference_df_dir:
        reference_df_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "소음원별 시나리오 v1.0.xlsx")
        info_df = pd.read_excel(reference_df_dir, usecols=[0,1,2,3])
    else:
        info_df = pd.read_excel(reference_df_dir)
    return pd.merge(info_df, target_df, how='outer', right_index=True, left_on='class ID')


def classID_to_korean(original_matrix: pd.DataFrame):
    """B1g -> g. 발전기의 가동소리, A3b -> b. 샤워할때 물소리 등으로 바꾼다. """
    matrix = original_matrix.copy()
    cols_to_rename = list(matrix.columns)
    merged = info_df_merge(matrix)
    mapper = { merged.loc[idx, "class ID"]:merged.loc[idx, "class상세"]
               for idx in merged.index if merged.loc[idx, "class ID"] in cols_to_rename}
    return matrix.rename(columns=mapper, index=mapper)


def df_normalized_by_row_net(df: pd.DataFrame):
    """dataframe의 행별 총합(TP+FN)으로 해당 행에 대한 셀값(pred)을 나눈 후 
    전체 셀에 대해 round(x, 4)*100을 적용한 데이터프레임을 리턴"""
    for idx in df.index:
        row_net = np.sum(list(df.loc[idx, :].values))
        df.loc[idx, :] = df.loc[idx, :]/row_net
    # return df.apply(lambda x: round(x, 4)*100)
    return df

def korean_rcParams(font_name: str="nanum"):
    # 폰트의 기본값들을 다음과 같이 설정할 수 있다
    plt.rcParams["font.family"] = "NanumMyeongjo"
    # plt.rcParams['font.size'] = 12.
    # plt.rcParams['xtick.labelsize'] = 24.
    # plt.rcParams['ytick.labelsize'] = 24.
    # 레이블에 유니코드의 '-'문자를 깨짐을 방지하기 위해 'axes.unicode_minus' 옵션을 False
    # plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['axes.labelsize'] = 20.
    # plt.rcParams["figure.figsize"] = (14,4)
    
    data = np.random.randint(-100, 100, 50).cumsum()
    plt.plot(range(50), data, 'r')
    plt.title('가격변동 추이')
    plt.ylabel('가격')
    plt.savefig( os.path.join(SAVE_PATH, "korean_rcParams.png"), bbox_inches='tight', dpi=100 )
    plt.show()
    plt.close()

def kor_fontprop(font_name: str="malgun.ttf"):
    font_paths = [ f for f in fm.findSystemFonts(fontpaths=None, fontext='ttf') if font_name in f ]

    if not font_paths:
        font_paths = [ f for f in fm.findSystemFonts(fontpaths=None, fontext='ttf') if "oding" in f ]
        print(f"Couldn't find any installed font including '{font_name}', so trying to search one including 'oding'.", *font_paths, sep="\n")
        if not font_paths:
            import subprocess
            # import locale
            # os_encoding = locale.getpreferredencoding()
            subprocess.call("fc-list :lang=ko | grep ttf", shell=True)
            print(f"Use one of these installed korean fonts above for font_name")

    return fm.FontProperties( fname=font_paths[0] )

def my_heatmap(data: pd.DataFrame, annot_size, annot_type="d", vmax_quantile=1, font_name="NanumMyeongjo.ttf", save_dir="./cf heatmap.png"):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (16,13))
    # # plt.figure(figsize = (16,8))
    # sns.set(rc={'figure.figsize':(11.7, 8.27)})
    # plt.gcf().subplots_adjust(bottom=0.1, left=0.3)
    # fig.subplots_adjust(left=0.1)
    yticks = data.index
    xticks = data.columns

    fontprop = kor_fontprop(font_name)
    ax.set_title("오차행렬", fontproperties=fontprop)
    ax.set_xlabel("y_pred", fontsize=14)
    ax.set_ylabel("y_true", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xticks(range(len(xticks)), xticks)
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_xticklabels(xticks, rotation=90, ha="right", fontproperties=fontprop)
    ax.set_yticklabels(yticks, rotation=0, ha="right", fontproperties=fontprop)
    # ax.set_yticklabels(yticks, rotation=45, ha="right", fontproperties=fontprop)

    mycolor = sns.diverging_palette(220, 20, as_cmap=True, ) # center="dark",
    # mycolor = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    # mycolor = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    # mycolor = sns.diverging_palette(250, 30, l=65, as_cmap=True)
    sns.heatmap(data, linewidth=0.5, annot=True, annot_kws={"size":annot_size}, fmt=annot_type, 
        cmap=mycolor, square=True, vmax=np.quantile(data.values, vmax_quantile)) #yticklabels=yticks, xticklabels=xticks, 

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight', dpi=100)
    plt.show()
    plt.pause(5)
    plt.close()


def final_report(matrix_dir, dataset_report_dir=False):
    original_matrix = pd.read_excel(matrix_dir, index_col=0)
    tmp, ext        = os.path.splitext(matrix_dir)
    save_path       = os.path.dirname(tmp)
    filename        = os.path.basename(tmp)
    result_file     = os.path.join( save_path, f"{filename} result{ext}" )
    summary_file    = os.path.join( save_path, f"{filename} summary{ext}" )
    heatmap_file    = os.path.join( save_path, f"{filename} heatmap.png" )
    heatmap_file2   = os.path.join( save_path, f"{filename} heatmap2.png" )

    # matrix.style.background_gradient(cmap='summer')
    matrix, summary = summarize_confusion_matrix(original_matrix)
    summary = info_df_merge(summary, reference_df_dir=dataset_report_dir)
    matrix.to_excel(result_file)
    summary.to_excel(summary_file)
    print("confusion matrix is saved at : ", summary_file)

    # 엑셀파일을 히트맵이미지로 변환
    mapped = classID_to_korean(original_matrix)
    mapped.to_excel(os.path.join(save_path, f"{filename} for heatmap{ext}"))

    # 한글 맵핑된 엑셀로 히트맵그리기
    # 우분투에 한글폰트 문제로 윈도우에서 작업하기 위해 주석처리
    my_heatmap(mapped, annot_size=10, annot_type="d", vmax_quantile=0.9737, save_dir=heatmap_file)
    mapped = df_normalized_by_row_net(mapped) # print(mapped.apply(lambda x: round(x, 4)*100))
    my_heatmap(mapped, annot_size=8, annot_type=".2f", vmax_quantile=0.9737, save_dir=heatmap_file2)


if __name__=="__main__":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH  = os.path.join( os.path.dirname(PROJECT_DIR), "saved_model" )
    matrix_dir = os.path.join( SAVE_PATH, "20221206 20-05-54 confusion.xlsx")
    # matrix_dir = "C:/home/NIA/data/20221129 11-11-53 confusion.xlsx"
    data_report_dir = os.path.join(PROJECT_DIR, "data", "data_report_2022-11-28 15-22-56.xlsx")
    # 한글설치가 안되어 있는 경우 설치된 곳에서 히트맵변환을 원하는 오차행렬경로와 함께 아래 코드를 실행함
    final_report(matrix_dir=matrix_dir, dataset_report_dir=data_report_dir)
