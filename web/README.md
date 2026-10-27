# FG-CLIP 图文监控系统 Web

基于FG-CLIP模型的RTSP流行人检测与文本布控系统。

## 功能特性

1. **设备管理** - 添加、编辑、删除RTSP流设备（IP、端口、用户名、密码）
2. **规则配置** - 设置文本布控规则和匹配分数阈值
3. **告警展示** - 展示满足条件的图片、时间、触发规则

## 安装

```bash
cd web
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

访问 http://localhost:5000

## 页面说明

- `/` - 首页，监控状态概览
- `/devices` - 设备管理
- `/rules` - 规则配置
- `/alerts` - 告警记录

## 配置

数据存储在 `web/data/` 目录：
- `devices.json` - 设备信息
- `rules.json` - 布控规则
- `alerts.json` - 告警记录
- `alerts_images/` - 告警图片