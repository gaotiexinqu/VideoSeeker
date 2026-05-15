# 将环境的 bin 目录强制插入到 PATH 最前面
export PATH=/mnt/tidal-alsh01/dataset/zeus/zhaoy/envs/verl/bin:$PATH

# 验证
which python
python --version

# 获取当前节点IP
CURRENT_IP=$(hostname -I | awk '{print $1}')

# 32卡
# HEAD_IP="10.144.201.133"

# 第一16卡
# HEAD_IP="10.144.202.17"

# 第二16卡
HEAD_IP="10.144.204.85"

# HEAD_IP="10.144.204.66"

RAY_PORT="30001"

echo "头节点IP: $HEAD_IP"
echo "当前节点IP: $CURRENT_IP"

# 判断节点角色
if [ "$CURRENT_IP" = "$HEAD_IP" ]; then
    echo ">>> 当前节点是头节点 <<<"
else
    echo ">>> 当前节点是工作节点 <<<"
fi


# 1. 先停掉所有 Ray 进程
ray stop --force

# 2. 清理 Ray 的临时文件
rm -rf /tmp/ray

# 3. 检查 /dev/shm（共享内存）是否正常
df -h /dev/shm
ls -la /dev/shm

# 4. 确认没有残留 Ray 进程
ps aux | grep ray

# 5. 查杀 Ray 端口
kill -9 $(lsof -t -i:$RAY_PORT 2>/dev/null) 2>/dev/null

# 6. 清理 SGLang 相关残留进程（SGLang 使用随机端口做 TCPStore rendezvous）
echo ">>> 清理 SGLang 相关残留进程 <<<"
for pid in $(ps aux | grep -E "sglang|SGLang|python.*sglang" | grep -v grep | awk '{print $2}'); do
    echo "Kill SGLang process: $pid"
    kill -9 $pid 2>/dev/null
done

# # 7. 清理常见的 SGLang/TCPStore 端口范围 (30000-65000)
# # 这个范围内的端口可能被之前的 SGLang TCPStore 占用
# echo ">>> 清理 SGLang 常用端口范围内的残留进程 <<<"
# _scanned=0
# _killed=0
# for port in $(seq 30000 1 60999); do
#     pid=$(lsof -t -i:$port 2>/dev/null)
#     _scanned=$((_scanned + 1))
#     if [ -n "$pid" ]; then
#         echo "[CHECK] 端口 $port 仍被占用，正在检查占用进程..."
#         # 只杀掉 python 进程（SGLang 服务通常是 python）
#         for p in $pid; do
#             proc_name=$(ps -p $p -o comm= 2>/dev/null)
#             if [[ "$proc_name" == "python"* ]]; then
#                 echo "[KILL] 杀 python 进程 $p on port $port ($proc_name)"
#                 kill -9 $p 2>/dev/null
#                 _killed=$((_killed + 1))
#             else
#                 echo "[SKIP] 跳过非 python 进程 $p on port $port ($proc_name)"
#             fi
#         done
#     fi
#     # 每 1000 个端口打印一次进度
#     if [ $((_scanned % 1000)) -eq 0 ]; then
#         echo "[PROGRESS] 已扫描端口 ${_scanned}/31000，被杀进程数: ${_killed}"
#     fi
# done
# echo ">>> 端口清理完成，共扫描 ${_scanned} 个端口，杀掉 ${_killed} 个 python 进程 <<<"

# 8. 清理 /tmp 下可能的 SGLang 残留文件
rm -rf /tmp/sglang* 2>/dev/null

# 9. 等待清理完成
sleep 2

# 10. 根据节点角色启动 Ray
if [ "$CURRENT_IP" = "$HEAD_IP" ]; then
    echo "===== 当前是头节点 ====="
    echo "启动 Ray 头节点..."
    ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT
else
    echo "===== 当前是工作节点 ====="
    echo "连接至头节点 $HEAD_IP:$RAY_PORT ..."
    ray start --address="$HEAD_IP:$RAY_PORT"
fi

# lsof -i :30000-60999 2>/dev/null || echo "lsof 不可用"

# # 查看 Ray 相关进程
# ps aux | grep -E "ray|sglang|verl" | grep -v grep

# # 查看所有 Python 进程
# ps aux | grep python | grep -v grep