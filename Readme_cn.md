如果项目对您有帮助，欢迎点击右上角 **Star** 支持！✨  
### **OnnxOCR**  
### ![onnx_logo](onnxocr/test_images/onnxocr_logo.png)  

**基于 ONNX 的高性能多语言 OCR 引擎**  
![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)  
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)  
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)  
![Python Version](https://img.shields.io/badge/python-≥3.6-blue.svg)  


## 🚀 版本更新  
- **2025.05.21**  
  1. 新增 PP-OCRv5 模型，单模型支持 5 种文字类型：简体中文、繁体中文、中文拼音、英文和日文。  
  2. 整体识别精度相比ppocrv4提升13个百分点
  3. 精度与Paddleocr3.0保持一致。


## 🌟 核心优势  
1. **脱离深度学习训练框架**：可直接用于部署的通用 OCR。  
2. **跨架构支持**：在算力有限、精度不变的情况下，使用 PaddleOCR 转成 ONNX 模型，重新构建的可部署在 ARM 架构和 x86 架构计算机上的 OCR 模型。  
3. **高性能推理**：在同样性能的计算机上推理速度加速。  
4. **多语言支持**：单模型支持 5 种文字类型：简体中文、繁体中文、中文拼音、英文和日文。  
5. **模型精度**：与 PaddleOCR 模型保持一致。
6. **国产化适配**：重构代码工程架构，只需简单进行推理引擎的修改，即可适配更多国产化显卡。



## 🛠️ 环境安装  
```bash  
python>=3.6  

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt  
```  

**注意**：  
- 默认使用 Mobile 版本模型，使用 PP-OCRv5_Server-ONNX 模型效果更佳。  
- Mobile 模型已存在于 `onnxocr/models/ppocrv5` 下，无需下载；  
- PP-OCRv5_Server-ONNX 模型过大，已上传至 [百度网盘](https://pan.baidu.com/s/1hpENH_SkLDdwXkmlsX0GUQ?pwd=wu8t)（提取码: wu8t），下载后将 `det` 和 `rec` 模型放到 `./models/ppocrv5/` 下替换即可。  


## 🚀 一键运行  
```bash  
python test_ocr.py  
```  


## 📡 API 服务（CPU 示例）  
### 启动服务  
```bash  
python app-service.py  
```  

### 测试示例  
#### 请求  
```bash  
curl -X POST http://localhost:5005/ocr \  
-H "Content-Type: application/json" \  
-d '{"image": "base64_encoded_image_data"}'  
```  

#### 响应  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "名称",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "标头",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🔧 二次开发接口说明（Extended APIs）

以下接口在 `OnnxOCR/app-service.py` 中实现，默认端口 `5005`：

### 新增 POST 接口
- `POST /ocr_image`：请求体 JSON `{"image": "<base64>"}`，返回标注图片（image/png）
- `POST /ocr_url`：请求体 JSON `{"url": "<http(s)图片地址>"}`，返回 OCR 结果 JSON
- `POST /ocr_url_image`：请求体 JSON `{"url": "<http(s)图片地址>"}`，返回标注图片（image/png）

示例：
```bash
curl -X POST http://localhost:5005/ocr_image \
  -H "Content-Type: application/json" \
  -d '{"image":"BASE64_DATA"}' --output result.png

curl -X POST http://localhost:5005/ocr_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/img.jpg"}'

curl -X POST http://localhost:5005/ocr_url_image \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/img.jpg"}' --output result.png
```

### 新增 GET 接口（URL 直链入参）
- `GET /url=<path>`：返回 OCR 结果 JSON（支持翻译）
- `GET /img=<path>`：返回图片（支持翻译后的渲染）

说明：
- 先调用 `GET /url=图片URL?key=<deepseek_key>` 将使用 DeepSeek 翻译并缓存结果；
- 随后可直接调用 `GET /img=图片URL`，服务会使用缓存中的译文，将原图上的文字替换为译文（覆盖绘制，不再显示右侧面板/置信度）。

示例：
```bash
# 1）获取 JSON 并触发 DeepSeek 翻译（带 key）
http://127.0.0.1:5005/url=https://example.com/img.jpg?key=sk_xxx

# 2）直接获取图片（不带 key，复用上一步缓存的译文）
http://127.0.0.1:5005/img=https://example.com/img.jpg
```

### DeepSeek 翻译
- 模型：`deepseek-chat`
- 接口：`https://api.deepseek.com/chat/completions`
- 传参：在 `GET /url=` 时通过 `key=sk_xxx` 传入 API Key 以启用翻译
- 返回：JSON 中为每条结果新增 `text_translated`；图片接口会直接将译文覆盖在原文位置（自动换行/动态字号，必要时扩展覆盖区域保证不截断）

### Prompt 配置
DeepSeek 的提示词读取顺序（优先级）：
1. URL 查询参数：`prompt=<本地文件路径>`
2. 环境变量：`DEEPSEEK_PROMPT_PATH` 或 `PROMPT_PATH`
3. 默认位置：项目根目录 `prompt.txt` 或 `OnnxOCR/prompt.txt`

示例：
```bash
http://127.0.0.1:5005/url=https://example.com/img.jpg?key=sk_xxx&prompt=D:\\GitHub\\KOOK_OCR\\prompt.txt
```

### 其他说明
- 中文绘制使用 PIL + `onnxocr/fonts/simfang.ttf`，避免中文乱码。
- 若未先使用 `GET /url=` 触发翻译缓存，`GET /img=` 将回退为仅显示 OCR 框与文本的原逻辑（无翻译）。
- 若需要自定义缓存策略（过期时间/持久化），可进一步扩展。


## 🐳 Docker 镜像环境（CPU）  
### 镜像构建  
```bash  
docker build -t ocr-service .  
```  

### 镜像启动  
```bash  
docker run -itd --name onnxocr-service-v3 -p 5006:5005 onnxocr-service:v3  
```  

### POST 请求  
```  
url: ip:5006/ocr  
```  

### 返回值示例  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "名称",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "标头",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🌟 效果展示  
| 示例 1 | 示例 2 |  
|--------|--------|  
| ![](result_img/r1.png) | ![](result_img/r2.png) |  

| 示例 3 | 示例 4 |  
|--------|--------|  
| ![](result_img/r3.png) | ![](result_img/draw_ocr4.jpg) |  

| 示例 5 | 示例 6 |  
|--------|--------|  
| ![](result_img/draw_ocr5.jpg) | ![](result_img/555.png) |  


## 👨💻 联系与交流  
### 求职信息  
本人正在寻求工作机会，欢迎联系！  
![微信二维码](onnxocr/test_images/myQR.jpg)  

### OnnxOCR 交流群  
#### 微信群  
![微信群](onnxocr/test_images/微信群.jpg)  

#### QQ 群  
![QQ群](onnxocr/test_images/QQ群.jpg)  




## 🎉 致谢  
感谢 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供的技术支持！  


## 🌍 开源与捐赠  
我热爱开源和 AI 技术，相信它们能为有需要的人带来便利和帮助，让世界变得更美好。如果您认可本项目，可以通过支付宝或微信进行打赏（备注请注明支持 OnnxOCR）。  

<img src="onnxocr/test_images/weixin_pay.jpg" alt="微信支付" width="200">
<img src="onnxocr/test_images/zhifubao_pay.jpg" alt="支付宝" width="200">


## 📈 Star 历史  
[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)  


## 🤝 贡献指南  
欢迎提交 Issues 和 Pull Requests，共同改进项目！  
