# 书生·浦语大模型Demo体验

## 环境准备

打开jupyterlab的终端，克隆已有环境，自定义环境命名
```
cd /root
conda create --name internlm_antonia --clone /root/share/install_conda_env_internlm_base.sh internlm-demo
```
激活环境
```
Conda activate internlm_antonia
```
### 安装python相关依赖

```bash
# 升级pip
python -m pip install --upgrade pip
# 安装相关依赖
pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```
### 模型下载

平台的 `share` 目录下已经为我们准备了全系列的 `InternLM` 模型，所以我们可以直接复制。使用如下命令复制：

```shell
mkdir -p /root/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory
```
## InternLM-Chat-7B demo 实现

### 代码准备

首先 `clone` 代码，在 `/root` 路径下新建 `code` 目录，然后切换路径, clone 代码.
```shell
cd /root/code
git clone https://gitee.com/internlm/InternLM.git
```
切换 commit 版本

```shell
cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17
```

将 `/root/code/InternLM/web_demo.py` 中 29 行和 33 行的模型更换为本地的 `/root/model/Shanghai_AI_Laboratory/internlm-chat-7b`。

### 终端运行

在 `/root/code/InternLM` 目录下新建一个 `cli_demo.py` 文件

```bash
cd /root/code/InternLM
touch cli_demo.py && vim cli_demo.py
```

将以下代码填入其中：
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

然后在终端运行以下命令，即可体验 `InternLM-Chat-7B` 模型的对话能力

```shell
python /root/code/InternLM/cli_demo.py
```

### web demo 运行

切换到 `VScode` 中，运行 `/root/code/InternLM` 目录下的 `web_demo.py` 文件，输入以下命令后

```shell
bash
conda activate internlm_antonia
cd /root/code/InternLM
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```
网页版对话，生成的故事创作结果如下：

## Lagent 智能体 Demo 实现



