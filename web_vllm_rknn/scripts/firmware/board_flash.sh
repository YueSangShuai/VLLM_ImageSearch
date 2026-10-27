#!/bin/bash
# Flash new boot-chain partitions (dtbo/resource/kernel/boot) + install kernel 6.1.141 modules.
# Preconditions (checked here): backups exist at /data/part-backup, new images at /home/pi/firmware.
set -e
FW=/home/pi/firmware
BK=/data/part-backup
PL=/dev/disk/by-partlabel

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
for f in dtbo.img resource.img kernel.img boot.img; do
  [ -s $BK/$f ] || { echo "MISSING BACKUP $BK/$f - abort"; exit 1; }
  [ -s $FW/$f ] || { echo "MISSING NEW IMAGE $FW/$f - abort"; exit 1; }
done

echo "== verify backup integrity (partition content == backup)"
for p in dtbo resource kernel boot; do
  a=$(dd if=$PL/$p bs=1M 2>/dev/null | sha256sum | cut -d' ' -f1)
  b=$(sha256sum $BK/$p.img | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "backup mismatch for $p, abort"; exit 1; }
done

echo "== write new images"
for p in dtbo resource kernel boot; do
  dd if=$FW/$p.img of=$PL/$p bs=1M conv=fsync status=none
  echo "  $p written"
done
sync

echo "== readback verify"
for p in dtbo resource kernel boot; do
  a=$(dd if=$PL/$p bs=1M 2>/dev/null | head -c $(( $(stat -c%s $FW/$p.img) )) | sha256sum | cut -d' ' -f1)
  b=$(sha256sum $FW/$p.img | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "!!! READBACK MISMATCH $p - restoring backup"; dd if=$BK/$p.img of=$PL/$p bs=1M conv=fsync status=none; exit 1; }
  echo "  $p verified"
done

echo "== install kernel 6.1.141 modules"
if [ -s /home/pi/modules-6.1.141.tar.gz ] && [ ! -d /lib/modules/6.1.141 ]; then
  tar xzf /home/pi/modules-6.1.141.tar.gz -C /lib/modules/
  depmod -a 6.1.141 || true
  ls /lib/modules/6.1.141/modules.dep.bin >/dev/null && echo "  modules ok"
fi

echo "== ALL DONE, safe to reboot"
