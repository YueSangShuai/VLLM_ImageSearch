import json, subprocess, urllib.parse

d = json.load(open("/home/web_siglip_reid_rknn/data/devices.json"))[0]
path = d.get("main_stream_path", "/ch01") or "/ch01"
url = "rtsp://{}:{}@{}:{}{}".format(
    urllib.parse.quote(str(d.get("username", ""))),
    urllib.parse.quote(str(d.get("password", ""))),
    d.get("ip"), d.get("port", "554"), path)
r = subprocess.run(
    ["ffmpeg", "-rtsp_transport", "tcp", "-i", url, "-frames:v", "1",
     "-y", "/tmp/probe.jpg"], capture_output=True, text=True, timeout=25)
import os
print("ffmpeg rc:", r.returncode, "jpg size:", os.path.getsize("/tmp/probe.jpg") if os.path.exists("/tmp/probe.jpg") else 0)
print(r.stderr[-200:] if r.returncode else "ok")
