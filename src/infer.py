# infer Llama-3.2-1B

# conda activate edge

import os, json
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def format_chat_template(batch, tokenizer):

    system_prompt =  """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Llama-3.2-1B model.")
    parser.add_argument("--model_path", type=str, default="/nas2/checkpoints/Llama-3.2-1B")
    parser.add_argument("--quant", action="store_true") # default=False
    # parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input", type=str, default="Hello, how are you?")
    args = parser.parse_args()


    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.quant:
        raise NotImplementedError("Quantized loading is not implemented yet.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,     
            device_map="auto",
        )

    # Tokenize input
    inputs = tokenizer(args.input, return_tensors="pt").to(model.device)

    # Generate output
    model.eval()
    with torch.no_grad():
        # outputs = model.generate(**inputs, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 필요에 따라 True/False
            # The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
        )

    # Decode and print output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Inference Result:")
    print(result)