# set -x

ulimit -n 65535

cd /mnt/tidal-alsh01/dataset/zeus/zhaoy/videoseeker/verl

wandb login "[token]"
export WANDB_MODE=online
# export FORCE_QWENVL_VIDEO_READER="torchcodec"

PROJECT_DIR="/mnt/tidal-alsh01/dataset/zeus/zhaoy/videoseeker/verl"
CONFIG_PATH="$PROJECT_DIR/examples/video_tools/config"
CKPT_DIR=$PROJECT_DIR/checkpoints
PROJECT_NAME="videoseeker"
EXPERIMENT_NAME="experiment_name"

# MODEL_PATH 
MODEL_PATH="/path/to/model"

TRAIN_DATA_PATH="/path/to/train_data.parquet"
VAL_DATA_PATH="/path/to/val_data.parquet"
TOOL_CONFIG_PATH="$PROJECT_DIR/examples/video_tools/config/mcp_tool_config_1tool.yaml"

export DEBUG_MODE="False"
export VERL_LOGGING_LEVEL="WARN"

mkdir -p "$CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='timer1_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.default_local_dir=$CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.save_freq=50 \
    trainer.test_freq=1000 \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=$CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME/rollout \
    custom_reward_function.path=$PROJECT_DIR/custom_rewards/reward.py \
    +custom_reward_function.reward_kwargs.use_iou_reward=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    2>&1 | tee $CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME/training_log.log