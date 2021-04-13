import os
from understanding_rl_vision import rl_clarity
try:
    ON_CLUSTER = os.environ['CLUSTERNAME'] == 'leonhard'
except KeyError:
    ON_CLUSTER = False

if ON_CLUSTER:
    checkpoint_path = '/cluster/home/dlauro/aisc2021/understanding-rl-vision/checkpoints/coinrun_aisc.jd'
    output_dir = '/cluster/home/dlauro/aisc2021/understanding-rl-vision/outputs/'
else:
    checkpoint_path = '/home/lauro/code/aisc2021/understanding-rl-vision/checkpoints/coinrun_aisc.jd'
    output_dir='/tmp/'


# script runs faster if we ignore the other layers
layer_kwargs={
        'name_contains_one_of': ['2b'],
        }


load_kwargs={
        "coinrun_aisc": True,
        }


rl_clarity.run(checkpoint_path,
               output_dir=output_dir,
               layer_kwargs=layer_kwargs,
               load_kwargs=load_kwargs)
