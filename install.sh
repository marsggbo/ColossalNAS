# install colossalai
cd /workspace && \
git clone https://github.com/hpcaitech/ColossalAI.git \
&& cd ./ColossalAI \
&& CUDA_EXT=1 pip install -v --no-cache-dir .

# install titans
cd /workspace && \
pip install --no-cache-dir titans

# install tensornvme
cd /workspace && \
conda install cmake && \
git clone https://github.com/hpcaitech/TensorNVMe.git && \
cd TensorNVMe && \
pip install -r requirements.txt && \
pip install -v --no-cache-dir .

# install hyperbox
cd /workspace && \
git clone https://github.com/marsggbo/hyperbox.git && \
cd hyperbox && \
pip install -r requirements.txt && \
python setup.py develop

pip install pytest, transformers