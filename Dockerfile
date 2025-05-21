# 使用 Python 3.7 作为基础镜像
FROM python:3.7-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 到工作目录
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1

# 复制项目目录中的所有文件到镜像中
COPY . .

# 设置环境变量（如果需要）
ENV PYTHONUNBUFFERED=1

# 暴露服务端口（假设你的 Flask 服务运行在 5005 端口）
EXPOSE 5005

# 启动 Flask 服务
CMD ["python", "app-service.py"]