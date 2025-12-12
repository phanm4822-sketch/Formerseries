from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# 1. 指定模型 ID（就是你发的那个 ModelScope 模型）
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# 2. 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",       # 自动选择 float16/bfloat16 等
    device_map="auto",  # 自动把模型放到 GPU 或 CPU
)

# 3. 加载多模态 processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 4. 构造一条多模态对话（用官方示例图片的 URL）
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "用中文详细描述一下这张图片。",
            },
        ],
    }
]

# 5. 把对话转成模型输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

# 把张量移到模型所在设备
inputs = inputs.to(model.device)

# 6. 推理：生成回复
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 7. 只保留“新生成”的回答部分
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 8. 解码成文本
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("====== Qwen3-VL-2B-Instruct 输出 ======")
print(output_text[0])
print("======================================")
