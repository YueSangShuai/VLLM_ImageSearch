#!/usr/bin/env python3
"""Helper: run a command on the NanoPi R6S board via password SSH (no sshpass on host)."""
import sys, pexpect

HOST, USER, PWD = "10.8.18.118", "pi", "pi"

def run(cmd, timeout=60):
    ssh = pexpect.spawn(f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {USER}@{HOST} {cmd!r}",
                        encoding="utf-8", timeout=timeout)
    i = ssh.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
    if i == 0:
        ssh.sendline(PWD)
        ssh.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    out = ssh.before
    ssh.close()
    return out, (ssh.exitstatus if ssh.exitstatus is not None else -1)

if __name__ == "__main__":
    timeout = float(sys.argv[1]) if sys.argv[1].replace(".", "").isdigit() else 60
    cmd = sys.argv[-1]
    out, rc = run(cmd, timeout)
    print(out)
    sys.exit(rc)
