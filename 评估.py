import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm

# 用于正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示符号
plt.rcParams['axes.unicode_minus'] = False

mm.lap.default_solver = 'lap'

gt_file = "./gt.txt"

deep_ts_file = "./deep_ts.txt"


gt = mm.io.loadtxt(gt_file, fmt="mot16", min_confidence=1)

deep_ts = mm.io.loadtxt(deep_ts_file, fmt="mot16")



deep_ts = deep_ts.sort_values(by=["Id", "FrameId"])



deep_acc = mm.utils.compare_to_groundtruth(gt, deep_ts, 'iou', distth=0.5)


mh = mm.metrics.create()
metrics = ['num_frames', 'num_switches', 'idp', 'idr', 'idf1', 'mota', 'motp', 'precision', 'recall']

deep_summary = mh.compute(deep_acc, metrics=metrics, name='deepsort')

summary = pd.concat([deep_summary], axis=0, join='outer', ignore_index=False)

if os.path.exists("result.csv"):
    os.remove("result.csv")
summary.to_csv("result.csv")

