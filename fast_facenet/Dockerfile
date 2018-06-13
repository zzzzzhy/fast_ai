FROM solderzzc/rocketchat:mxnet

RUN apt-get update && apt-get install -y  python-sklearn python-skimage

ADD ./deploy /root/deploy
ADD ./model-y1-test2 /root/model-y1-test2
WORKDIR /root/deploy
RUN pip install -r requirements.txt
RUN python test.py --model=../model-y1-test2/model,0
