cd /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl

bash scripts/install_vllm_sglang_mcore.sh 2>&1 | tee scripts/install.log

pip install wandb==0.23.1

pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless
python -c "import cv2; print(cv2.__version__)"

pip install qwen-vl-utils==0.0.14

pip install torchcodec==0.7.0
conda install -c conda-forge ffmpeg
conda list ffmpeg

pip install "sglang[all]==0.5.2" --no-cache-dir

# install flash_attn
pip install --no-cache-dir /mnt/tidal-alsh01/dataset/zeus/zhaoy/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python /mnt/tidal-alsh01/dataset/zeus/zhaoy/tests/test_flash_attn.py

pip install uvicorn==0.40.0
pip install starlette==0.50.0

cd /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl
pip install --no-deps -e . 2>&1 | tee -a scripts/install.log

pip install transformers==4.57.3

pip install requests==2.32.5
pip install urllib3==2.6.3
pip uninstall chardet -y