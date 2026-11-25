#!/usr/bin/env python3
"""板子失联时用：扫给定 IP 列表，找 pi/pi 可登录且是 aarch64 / 有 /home/web_vllm_rknn 的机器。
用法: python3 find_board.py [ip.ip.ip.ip ...]（缺省扫 10.8.18.0/24 里 ping 通的）"""
import sys, pexpect, subprocess

CAND = sys.argv[1:]
if not CAND:
    r = subprocess.run(
        "seq 1 254 | xargs -P64 -I{} sh -c 'ping -c1 -W1 -w2 10.8.18.{} >/dev/null 2>&1 && echo 10.8.18.{}'",
        shell=True, capture_output=True, text=True, timeout=120)
    CAND = r.stdout.split()

def probe(ip):
    cmd = (f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
           f"-o ConnectTimeout=5 -o PreferredAuthentications=password "
           f"-o PubkeyAuthentication=no pi@{ip} "
           f"'uname -m; hostname; ls -d /home/web_vllm_rknn 2>/dev/null'")
    sp = pexpect.spawn(cmd, encoding="utf-8", timeout=12)
    i = sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if i == 0:
        sp.sendline("pi")
        sp.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=12)
    out = (sp.before or "").strip()
    sp.close()
    return out

hits = []
for ip in CAND:
    out = probe(ip)
    low = out.lower()
    if "aarch64" in out or "web_vllm_rknn" in out:
        hits.append((ip, out))
        print(f"*** 候选命中 {ip}: {out}")
    elif "denied" in low:
        print(f"{ip}: 密码不对")
    elif out:
        print(f"{ip}: {out[:100]}")

if not hits:
    print("未找到：/24 内 pi/pi 可登录的机器里没有 aarch64/部署目录。板子大概率关机/死机/在别的网段。")
