from transformers import pipeline

pipe = pipeline(model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
res = pipe("I have a problem with my sales that needs to be resolved asap!",
    candidate_labels=["sales", "not sales"],
)
print(res)