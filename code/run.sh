#!/bin/bash
#SBATCH --job-name=ner-vllm
#SBATCH --account=project_2005072
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:4
#SBATCH --output=ner_vllm_out.log
#SBATCH --error=ner_vllm_err.log

module load pytorch/2.6

# 设置 HuggingFace 模型缓存路径
export HF_HOME=/scratch/project_2005072/keshu/hf-cache

# 设置日志输出路径
VLLM_LOG=/scratch/project_2005072/keshu/vllm-logs/${SLURM_JOB_ID}.log
mkdir -p $(dirname $VLLM_LOG)

# 启动 vLLM 模型服务（后台运行）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 4 \
    --dtype half \
    --enforce-eager > $VLLM_LOG &

VLLM_PID=$!
echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# 等待服务启动（初始延时）
sleep 30

# 使用 curl 检查是否成功启动
while ! curl -s localhost:8000/v1 > /dev/null 2>&1
do
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        echo "❌ vLLM crashed during startup. Check log: $VLLM_LOG"
        exit 1
    fi
    echo "⌛ Waiting for vLLM to become available..."
    sleep 10
done

echo "✅ vLLM is ready. Running inference..."

# 启动推理脚本
python /scratch/project_2005072/keshu/cascade-camp-hands-on/code/main.py

# 推理结束后关闭服务
kill $VLLM_PID
echo "✅ Inference completed. vLLM stopped."
