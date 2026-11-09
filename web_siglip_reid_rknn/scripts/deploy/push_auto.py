#!/usr/bin/env python3
"""push_full.sh 的免交互版：pexpect 自动输密码。usage: push_auto.py [DEST]"""
import sys, os, pexpect

HERE = os.path.dirname(os.path.abspath(__file__))
cmd = ["bash", os.path.join(HERE, "push_full.sh")] + sys.argv[1:]
sp = pexpect.spawn(" ".join(cmd), encoding="utf-8", timeout=3600)
i = 0
while i == 0:
    i = sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if i == 0:
        sp.sendline("pi")
print(sp.before[-800:] if sp.before else "")
sp.close()
print(f"exit={sp.exitstatus}")
sys.exit(0 if sp.exitstatus == 0 else 1)
