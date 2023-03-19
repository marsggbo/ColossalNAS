# install colossalai
cd /workspace && \
git clone https://github.com/hpcaitech/ColossalAI.git \
&& cd ./ColossalAI \
&& NO_CUDA_EXT=1 pip install -v --no-cache-dir .

# # install titans
# cd /workspace && \
# pip install --no-cache-dir titans

# install tensornvme
cd /workspace && \
apt update && \
apt install tmux libaio1 libaio-dev && \
git clone https://github.com/hpcaitech/TensorNVMe.git && \
cd TensorNVMe && \
pip install -r requirements.txt && \
DISABLE_URING=1 pip install -v --no-cache-dir .

# install hyperbox
cd /workspace && \
git clone https://github.com/marsggbo/hyperbox.git && \
cd hyperbox && \
pip install -r requirements.txt && \
python setup.py develop

pip install pytest transformers
pip install -U numpy