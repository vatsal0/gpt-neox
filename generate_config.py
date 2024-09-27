import json
import argparse
from collections import OrderedDict
import random
import os

# python gpt-neox/generate_config.py -n 32 -k 2 -r expert_prob_approx --aux_loss 0.01 --lr 3e-4 --name expertapprox32c2 --job_script
config = OrderedDict()

parser = argparse.ArgumentParser()

parser.add_argument('--num_experts', '-n', type=int, required=True)
parser.add_argument('--top_k', '-k', type=int, required=True)
parser.add_argument('--router', '-r', type=str, choices={'topk', 'expert_prob_approx', 'dense'}, required=True)
parser.add_argument('--dense_warmup', type=int, default=500)
parser.add_argument('--aux_loss', type=float, default=0.01)
parser.add_argument('--z_loss', type=float, default=0)
parser.add_argument('--lr', type=float, required=True)

parser.add_argument('--tokenizer', type=str, choices={'LLaMA', 'OLMoE', 'gemma'}, default='LLaMA')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--checkpoint', type=bool, default=True)

parser.add_argument('--job_script', action='store_true', default=False)

CHECKPOINT_DIR = '/lus/eagle/projects/DemocAI/vatsalb/'
DATA_PATH = 'data/fineweb/sample-10BT-tokenized-{tokenizer}/fineweb-train-{i:05d}-of-00102_text_document'
TOKENIZER_PATHS = {
  'LLaMA': '/lus/eagle/projects/DemocAI/vatsalb/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/tokenizer.json',
  'OLMoE': '/lus/eagle/projects/DemocAI/vatsalb/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/989ab78e3bfb4f7bb843d269387dfd477693c3db/tokenizer.json',
  'gemma': '/lus/eagle/projects/DemocAI/vatsalb/huggingface/hub/models--google--gemma-7b/snapshots/ff6768d9368919a1f025a54f9f5aa0ee591730bb/tokenizer.json'
}

with open('gpt-neox/configs/model.yml') as f:
  model_config = json.load(f)
  config.update(model_config)

with open('gpt-neox/configs/optimization.yml') as f:
  optim_config = json.load(f)
  config.update(optim_config)

args = parser.parse_args()

port = random.randint(49152, 65536)
config['master_port'] = port

config['moe_num_experts'] = args.num_experts
config['intermediate_size'] = 22528//args.num_experts
config['moe_top_k'] = args.top_k
config['moe_router_type'] = args.router
if args.router == 'dense':
  config['dense_warmup_iters'] = args.dense_warmup

config['moe_aux_loss_coeff'] = args.aux_loss
config['moe_z_loss_coeff'] = args.z_loss

config['optimizer']['params']['lr'] = args.lr
config['min_lr'] = args.lr / 10

if args.checkpoint:
  config['save'] = os.path.join(CHECKPOINT_DIR, args.name)
  config['load'] = os.path.join(CHECKPOINT_DIR, args.name)

config['train-data-paths'] = [DATA_PATH.format(tokenizer=args.tokenizer, i=i) for i in range(101)]
config['valid-data-paths'] = [DATA_PATH.format(tokenizer=args.tokenizer, i=101)]
config['test-data-paths'] = [DATA_PATH.format(tokenizer=args.tokenizer, i=101)]
config['vocab_file'] = TOKENIZER_PATHS[args.tokenizer]

with open(f'gpt-neox/configs/generated/{args.name}.yml', 'w') as f:
  f.write(json.dumps(config, indent=4).replace('1e-08', '1.0e-8'))

job_script = '''#!/bin/bash
#PBS -N moe_{name}
#PBS -l filesystems=home:eagle
#PBS -l select=16
#PBS -l walltime=03:00:00
#PBS -q prod
#PBS -A DemocAI

NODES_ARRAY=($(cat "${{PBS_NODEFILE}}" | sort | uniq))
HEAD_NODE=${{NODES_ARRAY[0]}}
HEAD_NODE_IP=$(getent hosts $HEAD_NODE | awk 'NR==1 {{ print $1 }}')
HOST_LIST=$(IFS=,; echo "${{NODES_ARRAY[*]}}")
NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NGPUS=$((NGPUS_PER_NODE * NNODES))
NCPUS_PER_GPU=8

export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT={port}
export WORLD_SIZE=$NGPUS  #should be 8
export NCCL_DEBUG=INFO

rm -f /lus/eagle/projects/DemocAI/vatsalb/hostfile/{name}
sed -e 's/$/ slots=4/' $PBS_NODEFILE > /lus/eagle/projects/DemocAI/vatsalb/hostfile/{name}
export DLTS_HOSTFILE=/lus/eagle/projects/DemocAI/vatsalb/hostfile/{name}
echo $DLTS_HOSTFILE 
export DEEPSPEED_TIMEOUT=2

echo "PATH=${{PATH}}" > .deepspeed_env
echo "LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}" >> .deepspeed_env
echo "http_proxy=${{http_proxy}}" >> .deepspeed_env
echo "https_proxy=${{https_proxy}}" >> .deepspeed_env

PORT={port}
if lsof -i :$PORT; then
  echo "Port $PORT is in use. Killing the process using the port."
  lsof -ti :$PORT | xargs kill -9
else
  echo "Port $PORT is free."
fi

echo "PBS_NODEFILE = $(cat ${{PBS_NODEFILE}})"
echo "NODES_ARRAY = ${{NODES_ARRAY[@]}}"
echo "HEAD_NODE = ${{HEAD_NODE}}"
echo "HEAD_NODE_IP = ${{HEAD_NODE_IP}}"
echo "HOST_LIST = ${{HOST_LIST}}"
echo "NNODES = ${{NNODES}}"
echo "NGPUS = ${{NGPUS}}"
echo "NCPUS_PER_GPU = ${{NCPUS_PER_GPU}}"
echo "MASTER_ADDR:MASTER_PORT = $MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $((NGPUS_PER_NODE - 1)))

cd gpt-neox
rm -f core.*
python deepy.py train.py -d configs/generated {name}.yml
'''
if args.job_script:
  with open(f'scripts/{args.name}.sh', 'w') as f:
    f.write(job_script.format(port=port, name=args.name))