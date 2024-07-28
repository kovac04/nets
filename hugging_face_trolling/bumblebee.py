from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("cornflakes and yogurt?")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
