# stdlib dependencies
import socket
import subprocess
from collections import namedtuple
import time
from typing import List, Callable
import os
import signal
import sys
import pdb

# namedtuples
GpuInfo = namedtuple("GpuInfo", ["idx", "free_mbs", "total_mbs", "timestamp"])
GpuJobInfo = namedtuple("GpuJobInfo", ["cmd", "total_mbs_gpus"])

# global vars
launched_processes = []


# sigterm functions
def on_exit(signum, frame):
    global launched_processes
    for p in launched_processes:
        try:
            p.kill()
        except Exception as e:
            print(f"Process {p} killing not successful")
    sys.exit(0)


# networks
def find_free_ports(n: int = 1):
    ports, sockets = [], []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            ports.append(s.getsockname()[1])
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()
    return ports

# jobs
def launch_cmd(
    command: str,
    env_command: str = None,
    capture_output: bool = True,
    capture_errors: bool = True,
    use_async: bool = False,
    command_apply_functions : List[Callable] = None,
):
    if command_apply_functions is not None:
        for c_fn in command_apply_functions:
            command = c_fn(command)
    cmd_args = command.split(" ")
    run_fn = subprocess.Popen if use_async else subprocess.run
    try:
        env = os.environ.copy()
        if env_command:
            env_var, env_value = env_command.split("=", 1)
            env[env_var] = env_value
        return run_fn(
            cmd_args,
            bufsize=1,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_errors else subprocess.DEVNULL,
            text=True,
            universal_newlines=True,
            env=env,
        )
    except Exception as e:
        print(f"Command {command} failed with: {e}")
        return None


# gpus
def get_memory_gpus(mode: str = "nvidia", most_free_first: bool = True):
    supported_modes = ["nvidia"]
    assert mode in supported_modes, f"Mode {mode} not supported"
    cmd = "nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits"
    raw_gpus_info = launch_cmd(cmd, capture_output=True).stdout.strip().splitlines()
    gpus_info = [
        GpuInfo(
            idx=int(raw_gpus_info[i].split(",")[0]),
            free_mbs=int(raw_gpus_info[i].split(",")[1]),
            total_mbs=int(raw_gpus_info[i].split(",")[2]),
            timestamp=time.time(),
        )
        for i in range(len(raw_gpus_info))
    ]
    return (
        gpus_info
        if not most_free_first
        else sorted(gpus_info, key=lambda x: x.free_mbs, reverse=True)
    )


def schedule_cmd_gpus(
        command_list: List[GpuJobInfo], timeout_s: int = 60 * 60 * 24 * 7, sleep_s: int = 3, command_apply_functions : List[Callable] = None
):
    # strategy: tries to fit largest job in the min gpus
    global launched_processes
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    start_time = time.time()
    scheduled = [False for _ in range(len(command_list))]
    command_list = sorted(command_list, key=lambda x: x.total_mbs_gpus, reverse=True)
    gpus_memory = get_memory_gpus()
    assert command_list[0].total_mbs_gpus < sum(
        gpus_memory[i].total_mbs for i in range(len(gpus_memory))
    ), f"Job {command_list[0].cmd} with {command_list[0].total_mbs_gpus} can't be scheduled onto current visible gpus with total capacity {sum(gpus_memory[i].total_mbs for i in range(len(gpus_memory)))}"
    while time.time() - start_time < timeout_s:
        homogeneous_free_mbs = [
            (i + 1) * gpus_memory[i].free_mbs for i in range(len(gpus_memory))
        ]
        non_scheduled_indices = [
            i for i, is_scheduled in enumerate(scheduled) if not is_scheduled
        ]
        if len(non_scheduled_indices) == 0:
            all_done = True
            for p in launched_processes:
                exit_code = p.poll()
                if exit_code is None:
                    all_done = False
                elif exit_code == 1:
                    stdout, stderr = p.communicate()
                    print(stdout, stderr)
                    pdb.set_trace()
            if all_done:
                return 0

        for cmd_idx in non_scheduled_indices:
            gpu_list_idx = next(
                (
                    i
                    for i, value in enumerate(homogeneous_free_mbs)
                    if value > command_list[cmd_idx].total_mbs_gpus
                ),
                -1,
            )
            if gpu_list_idx != -1:
                cmd_pre = f"CUDA_VISIBLE_DEVICES={','.join([str(gpu.idx) for gpu in gpus_memory[:gpu_list_idx+1]])} "
                process = launch_cmd(
                    command=command_list[cmd_idx].cmd,
                    env_command=cmd_pre,
                    use_async=True,
                    command_apply_functions=command_apply_functions,
                )
                launched_processes.append(process)
                scheduled[cmd_idx] = True
                print(f"Scheduled {command_list[cmd_idx].cmd} into {cmd_pre}")
                print(f"Sleeping for {sleep_s} seconds")
                time.sleep(sleep_s)
                break
        gpus_memory = get_memory_gpus()

# apply functions
def fresh_port_mod_fn(cmd: str):
    port = find_free_ports()
    cmd_parts = cmd.split()
    cmd_parts.insert(cmd_parts.index("torch.distributed.launch") + 1, f"--master_port={port}")
    return " ".join(cmd_parts)
