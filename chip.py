"""选择本进程使用的 TPU 芯片；必须在 import jax 之前导入。

只打开本 host 的一颗芯片（环境变量 TOP_K_CHIP，默认 1），并让它的两个 TensorCore 各作为一个 device 出现，程序只用其中一个。不同芯片上的进程可以同时运行。
"""
import os

os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_VISIBLE_CHIPS'] = os.environ.get('TOP_K_CHIP', '1')
os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --deepsea_chip_config_name=legacy".strip()
