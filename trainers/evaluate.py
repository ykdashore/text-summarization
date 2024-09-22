from datasets import load_metric

def calculate_rouge(predictions, references):
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return result
