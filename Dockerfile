FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-1.0.0

WORKDIR /opt/ml/wxcode
COPY ./* ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple



WORKDIR /opt/ml/wxcode/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /opt/ml/wxcode