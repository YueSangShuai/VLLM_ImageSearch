#!/bin/bash
# backup boot-chain partitions of eMMC to /data/part-backup
set -e
cd /dev/disk/by-partlabel
for p in dtbo resource kernel boot; do
  [ -f /data/part-backup/$p.img ] || dd if=$p of=/data/part-backup/$p.img bs=1M
done
sha256sum /data/part-backup/*.img
df -h / | tail -1
