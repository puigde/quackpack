# Quackpack, scheduling utils
<img src="quackpack.jpg" alt="Alt text" width="100"/>

## Installing
Install the repo from source
```bash
git clone git@github.com:puigde/quackpack.git
pip install .
```

## Using
### Basic Scheduling:
The main function in quackpack is `schedule_cmd_gpus`, its main arguments are:
```python
def schedule_cmd_gpus
    command_list: List[GpuJobInfo], # List of commands to schedule
    timeout_s: int = 60 * 60 * 24 * 7, # Timeout for successful scheduling of commands
    sleep_s: int = 3, # Waiting time between successful scheduling of commands
```
GpuJobInfo contains a string with the job cmd and an integer representing the gpu memory capacity of a job.
```python
GpuJobInfo = namedtuple("GpuJobInfo", ["cmd", "total_mbs_gpus"])
```
Quackpack will try to fit the job memory in homogeneous chunks in the minimum amount of gpus possible and to the ones that are more free.

**Example:** scheduling ten experiments with 10k mbs each and a waiting time of 30 seconds between schedules with the default timeout (1w)
```python
import quackpack as qp
cmd_list = ["python experiment_{i}.py" for i in range(1,11)]
cmd_list = [qp.GpuJobInfo(it_cmd, 10*1000) for it_cmd in command_list]
qp.schedule_cmd_gpus(cmd_list, sleep_s=30)
```

*Other arguments:*
```python
def schedule_cmd_gpus
    ...
    command_apply_functions: List[Callable] = None, # list of functions that get cmd string
    # and an environment dict and return a modified cmd string for launchtime
    debug_on_crash: bool = False, # will open a python debugger if crashed
```
We can dynamically adapt our commands with `command_apply_functions`, for instance, when quackpack schedules into multi-gpu we might want to update the number of processes per node in our `torch.distributed.launch` command accordingly. This is done by looking at the `CUDA_VISIBLE_DEVICES` env var which quackpack sets when scheduling.
```python
def fresh_n_proc_per_node(cmd: str, env):
    return insert_string(
        cmd,
        "torch.distributed.launch",
        f"--nproc_per_node={len(env['CUDA_VISIBLE_DEVICES'].split(','))}",
    )
```
where insert string is a simple utility function that puts a string after another in the command
```python
def insert_string(total_str, previous_str, insert_str):
    total_str = total_str.split()
    total_str.insert(total_str.index(previous_str) + 1, insert_str)
    return " ".join(total_str)
```
Both functions are found in quackpack and can be called directly with `qp.<function>` other examples:
* `fresh_port_mod_fn`: adds '--master_port={port}' where port is a free port after 'torch.distributed.launch'

**Example:** scheduling a `torch.distributed.launch` with 100Gbs budget with dynamic number of processes and nodes.
```python
import quackpack as qp
cmd = "python -m torch.distributed.launch experiment_1.py --some_arg"
qp.schedule_cmd_gpus(
    command_list=[qp.GpuJobInfo(cmd, 100*1000)],
    command_apply_functions=[qp.fresh_port_mod_fn, qp.fresh_n_proc_per_node],
)
```

## Development
Install dev dependencies
```bash
pip install pre-commit black
```
Install pre-commit hooks
```bash
pre-commit install
```
