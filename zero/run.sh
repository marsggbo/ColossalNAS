git pull
rm -rf /workspace/colossalnas/zero/logs
ln -s /results /workspace/colossalnas/zero/logs
rm /lib/x86_64-linux-gnu/libcuda.so.1
ln -s /lib/x86_64-linux-gnu/libcuda.so.5* /usr/lib/x86_64-linux-gnu/libcuda.so.1
rm /lib/x86_64-linux-gnu/libnvidia-ml.so.1
ln -s /lib/x86_64-linux-gnu/libnvidia-ml.so.5* /lib/x86_64-linux-gnu/libnvidia-ml.so.1
cd /workspace/TensorNVMe
pip uninstall tensornvme
DISABLE_URING=1 pip install -v --no-cache-dir .
cd /workspace/colossalnas/zero
bash ./scripts/batch_benchmark.sh
