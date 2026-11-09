#!/usr/bin/env python3
"""Binary-safe scp to/from board. usage: scp_pi.py push <local...> | pull <remote...> [--dest DIR]"""
import sys, pexpect, subprocess

HOST, USER, PWD = "pi@10.8.18.118", "pi", "pi"

def main():
    mode = sys.argv[1]
    rest = sys.argv[2:]
    dest = "."
    if "--dest" in rest:
        i = rest.index("--dest"); dest = rest[i+1]; rest = rest[:i]+rest[i+2:]
    base = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    args = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if mode == "push":
        args += rest + [f"{USER}@10.8.18.118:{dest}"]
    else:
        args += [f"{USER}@10.8.18.118:{p}" for p in rest] + [dest]
    sp = pexpect.spawn(" ".join(args), encoding="utf-8", timeout=3600)
    i = sp.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if i == 0:
        sp.sendline(PWD)
    sp.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=3600)
    print(sp.before[-300:] if sp.before else "")
    sp.close()
    sys.exit(0 if sp.exitstatus == 0 else f"scp failed rc={sp.exitstatus}")

if __name__ == "__main__":
    main()
