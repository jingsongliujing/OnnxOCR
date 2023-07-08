# paddleocr模型转换成onnx模型后，利用ONNX模型进行推理
## 1、安装paddle2onnx
```angular2html
pip install paddle2onnx
```

## 2、下载paddleocr模型文件
```angular2html
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
```
## 3、解压模型文件
```angular2html
!tar -xvf /home/aistudio/onnx_pred/models/ch_ppocr_mobile_v2.0_cls_infer.tar
!tar -xvf /home/aistudio/onnx_pred/models/ch_ppocr_server_v2.0_det_infer.tar
!tar -xvf /home/aistudio/onnx_pred/models/ch_ppocr_server_v2.0_rec_infer.tar
```

## 4、将paddleocr模型转成onxx模型
```angular2html
paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./ch_ppocr_server_v2.0_rec.onnx \
--opset_version 11 \
--enable_onnx_checker True


paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./ch_ppocr_server_v2.0_det.onnx \
--opset_version 11 \
--enable_onnx_checker True


paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./ch_ppocr_mobile_v2.0_cls.onnx \
--opset_version 11 \
--enable_onnx_checker True
```

## 5、安装onnx
```angular2html
pip install onnx==1.14.0
pip install onnxruntime-gpu==1.14.1
```

## 6、模型推理
```angular2html
    import cv2
    model = ONNXPaddleOcr()

    img = cv2.imread('./1.jpg')

    # ocr识别结果
    result = model.ocr(img)
    print(result)
    
    # 画box框
    sav2Img(img, result)
```