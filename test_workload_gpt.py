import pytest
import subprocess

# Define parameter lists for parametrize

world_sizes = [16, 32]
tensor_model_parallel_sizes = [1, 2, 4, 8]
pipeline_model_parallels = [1, 2, 4, 8]
global_batches = [1024]
micro_batches = [1, 2]
seq_lengths = [4096]
epoch_nums = [1]
use_distributed_optimizers = [True] 
use_flash_attns = [True, False] 
swiglus = [True]

# 创建组合参数列表
combined_params = [
    ('GPT_7B', 'Megatron', 32, 4096, 32),
    ('GPT_13B', 'Megatron', 40, 5120, 40),
    ('GPT_22B', 'Megatron', 48, 6144, 64),
    ('GPT_175B', 'Megatron', 96, 12288, 96),
]

@pytest.mark.parametrize('model_name, frame, num_layers, hidden_size, num_attention_heads', combined_params)
@pytest.mark.parametrize('world_size', world_sizes)
@pytest.mark.parametrize('tensor_model_parallel_size', tensor_model_parallel_sizes)
@pytest.mark.parametrize('pipeline_model_parallel', pipeline_model_parallels)
@pytest.mark.parametrize('global_batch', global_batches)
@pytest.mark.parametrize('micro_batch', micro_batches)
@pytest.mark.parametrize('seq_length', seq_lengths)
@pytest.mark.parametrize('epoch_num', epoch_nums)
@pytest.mark.parametrize('use_distributed_optimizer', use_distributed_optimizers)
@pytest.mark.parametrize('use_flash_attn', use_flash_attns)
@pytest.mark.parametrize('swiglu', swiglus)
def test_run_workload(model_name, frame, world_size, tensor_model_parallel_size, pipeline_model_parallel,
                      global_batch, micro_batch, num_layers, seq_length, hidden_size, epoch_num,
                      use_distributed_optimizer, num_attention_heads, use_flash_attn, swiglu):
    command = [
        'python', '-m', 'workload_generator.AIOB_simAI_workload_generator',
        '--model_name', model_name, '--frame', frame, '--world_size', str(world_size),
        '--tensor_model_parallel_size', str(tensor_model_parallel_size), '--pipeline_model_parallel', str(pipeline_model_parallel),
        '--global_batch', str(global_batch), '--micro_batch', str(micro_batch), '--num_layers', str(num_layers),
        '--seq_length', str(seq_length), '--hidden_size', str(hidden_size), '--epoch_num', str(epoch_num),
    ]

    if use_distributed_optimizer:
        command.append('--use-distributed-optimizer')
    if use_flash_attn:
        command.append('--use_flash_attn')
    if swiglu:
        command.append('--swiglu')

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        assert result.returncode == 0, f"Command failed with output: {result.stderr}"
    except subprocess.CalledProcessError as e:
        assert False, f"An error occurred: {e.stderr}"