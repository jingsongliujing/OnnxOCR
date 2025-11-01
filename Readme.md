
English | [简体中文](./Readme_cn.md) |

### **OnnxOCR**  
### ![onnx_logo](onnxocr/test_images/onnxocr_logo.png)  

**A High-Performance Multilingual OCR Engine Based on ONNX**  

[![GitHub Stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social&label=Star&maxAge=3600)](https://github.com/jingsongliujing/OnnxOCR/stargazers)  
[![GitHub Forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social&label=Fork&maxAge=3600)](https://github.com/jingsongliujing/OnnxOCR/network/members)  
[![GitHub License](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)](https://github.com/jingsongliujing/OnnxOCR/blob/main/LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-%E2%89%A53.6-blue.svg)](https://www.python.org/)  


## 🚀 Version Updates  
- **2025.05.21**  
  1. Added PP-OCRv5 model, supporting 5 language types in a single model: Simplified Chinese, Traditional Chinese, Chinese Pinyin, English, and Japanese.  
  2. Overall recognition accuracy improved by 13% compared to PP-OCRv4.  
  3. Accuracy is consistent with PaddleOCR 3.0.  


## 🌟 Core Advantages  
1. **Deep Learning Framework-Free**: A universal OCR engine ready for direct deployment.  
2. **Cross-Architecture Support**: Uses PaddleOCR-converted ONNX models, rebuilt for deployment on both ARM and x86 architecture computers with unchanged accuracy under limited computing power.  
3. **High-Performance Inference**: Faster inference speed on computers with the same performance.  
4. **Multilingual Support**: Single model supports 5 language types: Simplified Chinese, Traditional Chinese, Chinese Pinyin, English, and Japanese.  
5. **Model Accuracy**: Consistent with PaddleOCR models.  
6. **Domestic Hardware Adaptation**: Restructured code architecture for easy adaptation to more domestic GPUs by modifying only the inference engine.  


## 🛠️ Environment Setup  
```bash  
python>=3.6  

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt  
```  

**Note**:  
- The Mobile version model is used by default; the PP-OCRv5_Server-ONNX model offers better performance.  
- The Mobile model is already in `onnxocr/models/ppocrv5` and requires no download;  
- The PP-OCRv5_Server-ONNX model is large and uploaded to [Baidu Netdisk](https://pan.baidu.com/s/1hpENH_SkLDdwXkmlsX0GUQ?pwd=wu8t) (extraction code: wu8t). After downloading, place the `det` and `rec` models in `./models/ppocrv5/` to replace the existing ones.  


## 🚀 One-Click Run  
```bash  
python test_ocr.py  
```  


## 📡 API Service (CPU Example)  
### Start Service  
```bash  
python app-service.py  
```  

### Test Example  
#### Request  
```bash  
curl -X POST http://localhost:5005/ocr \  
-H "Content-Type: application/json" \  
-d '{"image": "base64_encoded_image_data"}'  
```  

#### Response  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "Name",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "Header",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🔧 Extended APIs

These endpoints are implemented in `OnnxOCR/app-service.py` (default port `5005`):

### New POST endpoints
- `POST /ocr_image`: body JSON `{"image": "<base64>"}`; returns annotated image (image/png)
- `POST /ocr_url`: body JSON `{"url": "<http(s) image URL>"}`; returns OCR result JSON
- `POST /ocr_url_image`: body JSON `{"url": "<http(s) image URL>"}`; returns annotated image (image/png)

Examples:
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

### New GET endpoints (URL path style)
- `GET /url=<path>`: returns OCR JSON (translation supported)
- `GET /img=<path>`: returns image (renders with translations when available)

Notes:
- First call `GET /url=<imageURL>?key=<deepseek_key>` to translate via DeepSeek and cache the result.
- Then call `GET /img=<imageURL>`; the service uses cached translations to overwrite original text on the image (no sidebar/confidence).

Examples:
```bash
# 1) Get JSON and trigger DeepSeek translation (with key)
http://127.0.0.1:5005/url=https://example.com/img.jpg?key=sk_xxx

# 2) Get image (without key; uses cached translations)
http://127.0.0.1:5005/img=https://example.com/img.jpg
```

### DeepSeek translation
- Model: `deepseek-chat`
- Endpoint: `https://api.deepseek.com/chat/completions`
- How to enable: provide API key via `GET /url=` query `key=sk_xxx`
- Return: JSON adds `text_translated`; image endpoint overlays translations over original text (auto wrap/dynamic font size; expands overlay region when needed to avoid truncation)

### Prompt configuration
Prompt resolving priority:
1. URL query param: `prompt=<local file path>`
2. Environment variables: `DEEPSEEK_PROMPT_PATH` or `PROMPT_PATH`
3. Fallback: project root `prompt.txt` or `OnnxOCR/prompt.txt`

Example:
```bash
http://127.0.0.1:5005/url=https://example.com/img.jpg?key=sk_xxx&prompt=D:\\GitHub\\KOOK_OCR\\prompt.txt
```

### Additional notes
- Rendering uses PIL + `onnxocr/fonts/simfang.ttf` to avoid garbled CJK characters.
- If you don't call `GET /url=` first to warm the translation cache, `GET /img=` falls back to the plain OCR visualization (no translation).
- Custom cache strategies (TTL/persistence) can be added if needed.


## 🐳 Docker Image Environment (CPU)  
### Build Image  
```bash  
docker build -t ocr-service .  
```  

### Run Image  
```bash  
docker run -itd --name onnxocr-service-v3 -p 5006:5005 onnxocr-service:v3  
```  

### POST Request  
```  
url: ip:5006/ocr  
```  

### Response Example  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "Name",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "Header",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🌟 Effect Demonstration  
| Example 1 | Example 2 |  
|-----------|-----------|  
| ![](result_img/r1.png) | ![](result_img/r2.png) |  

| Example 3 | Example 4 |  
|-----------|-----------|  
| ![](result_img/r3.png) | ![](result_img/draw_ocr4.jpg) |  

| Example 5 | Example 6 |  
|-----------|-----------|  
| ![](result_img/draw_ocr5.jpg) | ![](result_img/555.png) |  


## 👨💻 Contact & Communication  
### Career Opportunities  
I am currently seeking job opportunities. Welcome to connect!  
![WeChat QR Code](onnxocr/test_images/myQR.jpg)  

### OnnxOCR Community  
#### WeChat Group  
![WeChat Group](onnxocr/test_images/微信群.jpg)  

#### QQ Group  
![QQ Group](onnxocr/test_images/QQ群.jpg)  


## 🎉 Acknowledgments  
Thanks to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for technical support!  


## 🌍 Open Source & Donations  
I am passionate about open source and AI technology, believing they can bring convenience and help to those in need, making the world a better place. If you recognize this project, you can support it via Alipay or WeChat Pay (please note "Support OnnxOCR" in the remarks).  

<img src="onnxocr/test_images/weixin_pay.jpg" alt="WeChat Pay" width="200">
<img src="onnxocr/test_images/zhifubao_pay.jpg" alt="Alipay" width="200">


## 📈 Star History  
[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)  


## 🤝 Contribution Guidelines  
Welcome to submit Issues and Pull Requests to improve the project together!  
