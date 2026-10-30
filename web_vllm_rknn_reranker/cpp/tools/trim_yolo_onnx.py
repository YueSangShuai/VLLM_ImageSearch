#!/usr/bin/env python3
# yolov8n-dyn.onnx 的 dynamic 导出带 6 个 graph output（5 个是中间张量残留）。
# 只保留 output0，其余 graph_outputs 清空后另存，供 rknn 转换。
import sys, onnx
from onnx import helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
keep = [o for o in m.graph.output if o.name == "output0"]
assert len(keep) == 1, [o.name for o in m.graph.output]
del m.graph.output[:]
m.graph.output.extend(keep)
# 初值 input 是 dynamic h/w；固定到 640x640 便于静态转换（这里只裁输出，shape 由转换脚本 input_size_list 定）
onnx.save(m, dst)
print("trimmed ->", dst, "outputs:", [o.name for o in m.graph.output])
