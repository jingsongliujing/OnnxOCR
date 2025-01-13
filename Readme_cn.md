# 如果项目对您有帮助，麻烦右上角点个Star呗
# OnnxOCR

#### 版本更新
PPOCRv4模型转onnx模型推理，精度和性能更高，推理速度比使用paddlepaddle框架快5倍

####   一.优势：
1.脱离深度学习训练框架，可直接用于部署的通用OCR。
2.在算力有限，精度不变的情况下使用paddleOCR转成ONNX模型，进行重新构建的一款可部署在arm架构和x86架构计算机上的OCR模型。
3.在同样性能的计算机上推理速度加速了4-5倍。

#### 二.环境安装
    python>=3.6

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

    此版本模型已经存在于onnxocr\models\ppocrv4下，无需下载，假如需要跑ppocrv2版本，需要下载rec模型超过了100M，github有限制，所以我上传到
[百度网盘,提取码: 125c](https://pan.baidu.com/s/1O1b30CMwsDjD7Ti9EnxYKQ )
     
    下载后放到./models/ch_ppocr_server_v2.0/rec/rec.onnx下

#### 三.一键运行

    python test_ocr.py

#### 四.API Service(cpu示例)
运行服务示例：

```
python app-service.py
```
#### 测试示例：
##### 请求：

```
curl -X POST http://localhost:5005/ocr \
-H "Content-Type: application/json" \
-d '{"image": "base64_encoded_image_data"}'
```

##### 响应：

```
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

#### 五.docker镜像环境（cpu）[GPU版本的待更新]
##### 镜像构建：
```
docker build -t ocr-service .
```
##### 镜像启动：
```
docker run -itd --name onnxocr-service-v3 -p 5006:5005 onnxocr-service:v3
```

##### post请求：
```
url: ip:5006/ocr
```

##### 返回值示例：
```
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

#### 效果展示

![Alt text](result_img/draw_ocr_1.jpg)

![Alt text](result_img/draw_ocr2.jpg)

![Alt text](result_img/draw_ocr3.jpg)

![Alt text](result_img/draw_ocr4.jpg)

![Alt text](result_img/draw_ocr5.jpg)

![Alt text](result_img/draw_ocr.jpg)
##
##
##
### 最近在求一份工作，或者您有什么建议可以加我的微信

![alt text](onnxocr/test_images/myQR.jpg)

#### OnnxOCR交流群
##### 微信群
![alt text](onnxocr/test_images/微信群.jpg)

##### QQ群
![alt text](onnxocr/test_images/QQ群.jpg)

#### 感谢PaddleOcr

https://github.com/PaddlePaddle/PaddleOCR


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)
