from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import torch
import sys
from pathlib import Path

# Allow imports from the watermark repo
sys.path.append(str(Path(__file__).resolve().parents[1] / "external_lm_watermarking"))

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def main():
    model_name = "facebook/opt-125m"
    device = "cpu"   # start with cpu for reliability on Mac

    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "Explain why renewable energy is important in one short paragraph."

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2.0,
        seeding_scheme="selfhash",
    )

    tokenized_input = tokenizer(prompt, return_tensors="pt")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    with torch.no_grad():
        output_tokens = model.generate(
            **tokenized_input,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            logits_processor=LogitsProcessorList([watermark_processor]),
        )

    generated_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
    output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print("\n=== Watermarked Output ===\n")
    print(output_text)

    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme="selfhash",
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )

    score_dict = watermark_detector.detect(output_text)

    print("\n=== Detection Result ===\n")
    for k, v in score_dict.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()