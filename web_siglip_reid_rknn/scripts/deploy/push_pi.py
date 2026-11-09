#!/usr/bin/env python3
"""output/ 整包推送板端 pi@10.8.18.118:/home/web_siglip_reid_rknn/。

流程：本地 tar → scp_pi 同款 pexpect 推 tar → 板上解包替换 → md5 抽查二进制。
用法：python3 scripts/deploy/push_pi.py [--target DIR]
"""
import os
import subprocess
import sys
import pexpect

HOST, USER, PWD = "pi@10.8.18.118", "pi", "pi"
SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET = "/home/web_siglip_reid_rknn"
TARBALL = "/tmp/web_siglip_reid_rknn.tar"


def ssh(cmd, timeout=600):
    sp = pexpect.spawn(f"ssh {SSH_OPTS} {HOST} {cmd!r}", encoding="utf-8", timeout=timeout)
    if sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]) == 0:
        sp.sendline(PWD)
        sp.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    out = sp.before or ""
    sp.close()
    return sp.exitstatus, out


def scp(src, dst, timeout=3600):
    sp = pexpect.spawn(
        f"scp {SSH_OPTS} {src} {USER}@10.8.18.118:{dst}", encoding="utf-8", timeout=timeout)
    if sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]) == 0:
        sp.sendline(PWD)
        sp.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    sp.close()
    return sp.exitstatus


def main():
    global TARGET
    args = sys.argv[1:]
    if "--target" in args:
        TARGET = args[args.index("--target") + 1]

    os.chdir(ROOT)
    print("== tar output/ ->", TARBALL)
    subprocess.run(["tar", "cf", TARBALL, "output"], check=True)
    size = os.path.getsize(TARBALL) / 1e6
    print(f"   {size:.0f} MB")

    rc, out = ssh(f"mkdir -p {TARGET}")
    assert rc == 0, out

    print("== scp tarball (可能需要几十秒到几分钟)")
    rc = scp(TARBALL, "/tmp/")
    assert rc == 0, "scp failed"

    print("== 板上解包")
    rc, out = ssh(f"tar xf /tmp/web_siglip_reid_rknn.tar -C {TARGET} --strip-components=1 && rm /tmp/web_siglip_reid_rknn.tar && ls {TARGET}", timeout=1200)
    print(out)
    assert rc == 0, out

    rc, out = ssh(f"md5sum {TARGET}/web_siglip_reid")
    print("board:", out.strip())
    h = subprocess.run(["md5sum", "output/web_siglip_reid"], capture_output=True, text=True)
    print("host :", h.stdout.strip())
    print("DONE. 板上启动: cd", TARGET, "&& bash run_web_rk.sh")


if __name__ == "__main__":
    main()
