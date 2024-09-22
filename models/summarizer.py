from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class SummarizerModel:
    def __init__(self, model_ckpt="google/pegasus-cnn_dailymail",tokenizer=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(self.device)

    def summarize(self, text, max_length=128, num_beams=8, length_penalty=0.8):
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt", padding="max_length")
        summaries = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length
        )
        return self.tokenizer.decode(summaries[0], skip_special_tokens=True)
