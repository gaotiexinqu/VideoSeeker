# local_dir 填写RL保存ckpt里最后一个step的路径的actor文件夹 eg,. xxx/actor
# target_dir 填写转化成hf形式的权重保存路径，最好不要和上面的路径重了

cd /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl

ROOT="/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/checkpoints/Thinking_V2P_Videos/0430_1731_qwen3-vl-8b-instruct_6k5_sft_qwen-thinking-distill_rl-4k1_oe_1tool_bs32/global_step_50"

python /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir $ROOT/actor \
  --target_dir $ROOT/model_hf_0430_1731_qwen3-vl-8b-instruct_6k5_sft_qwen-thinking-distill_rl-4k1_oe_1tool_bs32_global_step_50