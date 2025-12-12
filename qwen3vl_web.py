import os
import torch
import gradio as gr
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

# 1. 模型 ID（就是你之前用的那个）
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

print("正在加载模型，这一步会比较慢，只需要等一次...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",       # 自动选择精度
    device_map="auto",  # 自动放到 GPU / CPU，需要 accelerate
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("模型加载完成！")


def qwen3vl_infer(image_path, text_prompt):
    if image_path is None:
        return "请先上传一张图片。"
    if not text_prompt or text_prompt.strip() == "":
        return "请先输入一个问题（中文或英文都可以）。"

    # Gradio 传进来已经是一个本地文件路径，例如 C:\Users\...\OIP.jpg
    abs_path = os.path.abspath(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": abs_path},  # 直接传路径就行
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_texts[0]



# 3. 搭建 Gradio 界面
demo = gr.Interface(
    fn=qwen3vl_infer,
    inputs=[
        gr.Image(type="filepath", label="上传图片（本地文件）"),
        gr.Textbox(lines=3, label="问题 / 指令（例如：详细描述这张图片）"),
    ],
    outputs=gr.Textbox(label="Qwen3-VL 回答"),
    title="Qwen3-VL-2B-Instruct 本地多模态 Demo",
    description="上传一张图片，输入你的问题，点击提交即可获得回答。",
)

if __name__ == "__main__":
    # 这里可以改 server_port，比如 7861
    demo.launch(server_name="127.0.0.1", server_port=7860)
