#!/usr/bin/env python3
"""整包推送 output_cpp/ -> 板上 /home/web_vllm_rknn_q35（tar|ssh 管道，pexpect 自动输密码）。
用法: python push_full_py.py [--dest-base /home]；本地 output_cpp 内可用软链，-h 解引用。"""
import sys, os, pexpect

HOST, USER, PWD = "pi@10.8.18.118", "pi", "pi"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dest_base = "/home"
if "--dest-base" in sys.argv:
    dest_base = sys.argv[sys.argv.index("--dest-base") + 1]
REMOTE_DIR = f"{dest_base}/web_vllm_rknn_q35"

os.chdir(ROOT)
cmd = (f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {HOST} 'mkdir -p {REMOTE_DIR}' && "
       f"tar cf - -h output_cpp | "
       f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {HOST} 'tar xf - -C {REMOTE_DIR}'")
sp = pexpect.spawn("bash -c " + repr(cmd), encoding="utf-8", timeout=1800)
while True:
    i = sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if i == 0:
        sp.sendline(PWD)
    else:
        break
sp.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=1800)
sp.close()
print((sp.before or "")[-200:])
sys.exit(0 if sp.exitstatus == 0 else f"push failed rc={sp.exitstatus}")
