import json
p = "/home/web_siglip_reid_rknn/data/tasks.json"
t = json.load(open(p))
dev = json.load(open("/home/web_siglip_reid_rknn/data/devices.json"))[0]["id"]
for x in t:
    if not x.get("device_id"):
        x["device_id"] = dev
json.dump(t, open(p, "w"), ensure_ascii=False, indent=2)
print("bound tasks to device", dev)
