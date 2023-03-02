FROM python:3.8

RUN apt-get update && apt-get -y install sudo vim && \
    pip install torch torchvision torchaudio matplotlib==3.5.3 pandas==1.4.4 scikit-learn==1.1.2 scikit-image==0.19.3 seaborn==0.12.1 openpyxl==3.0.10 && \
    # 나눔글꼴(fonts-nanum)을 설치하고, fc-cache 명령으로 폰트 캐시 삭제
    apt-get -y install fonts-nanum* && fc-cache -fv && \
    # 나눔 글꼴을 matplotlib 에 복사하고, matplotlib의 폰트 캐시를 삭제 
    cp /usr/share/fonts/truetype/nanum/Nanum* /usr/local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/ && \
    rm -rf /home/ubuntu/.cache/matplotlib/*

ARG UNAME=testuser
ARG USERID=1000
ARG GROUPID=1000
RUN echo "Build arg UID=${USERID}"

RUN groupadd -g $GROUPID -o $UNAME
RUN useradd -m -u $USERID -g $GROUPID -o -s /bin/bash $UNAME
# host uid&user와 동일한 계정생성 및 권한부여
# RUN adduser --disabled-password --gecos "" $UNAME -u $USERID
# RUN adduser --disabled-password --gecos "" $UNAME --uid $USERID --gid $GROUPID
# RUN echo $UNAME:$UNAME | chpasswd
RUN adduser $UNAME sudo
RUN echo $UNAME 'ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN usermod -aG sudo $UNAME

RUN sudo mkdir -v --parent /mnt/data
RUN sudo mkdir -v /saved_model
RUN sudo mkdir -v /app
# 소유권 변경. 소유권이 있어야 허가권 변경가능
RUN sudo chown -R $UNAME:$UNAME /mnt/data
RUN sudo chown -R $UNAME:$UNAME /app/
RUN sudo chown -R $UNAME:$UNAME /saved_model/
# 현사용자의 허가권 변경. 디렉토리 허가권이 파일보다 우선시 된다..
RUN sudo chmod -R 777 /mnt/data
RUN sudo chmod -R 777 /app/
RUN sudo chmod -R 777 /saved_model/
# 데이터셋 배치작업 및 전처리 파일
COPY --chown=$USERID:$GROUPID ./sample_handler.py /app/sample_handler.py
COPY --chown=$USERID:$GROUPID ./custom_preprocessor.py /app/custom_preprocessor.py
# 학습관련 파일
COPY --chown=$USERID:$GROUPID ./data /app/data
COPY --chown=$USERID:$GROUPID ./torch_main_copy.py /app/torch_main_copy.py
COPY --chown=$USERID:$GROUPID ./run.py /app/run.py
COPY --chown=$USERID:$GROUPID ./f1score.py /app/f1score.py
# 유효성 검증 관련 파일
COPY --chown=$USERID:$GROUPID ./test_only.py /app/test_only.py

WORKDIR /app
USER $UNAME
CMD ["tail","-f","/dev/null"]