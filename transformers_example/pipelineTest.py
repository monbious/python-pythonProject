from transformers import pipeline
import torch
import torch.nn.functional as F

classifier = pipeline("sentiment-analysis")
results = classifier(["We are very happy to show you the Transformers library.",
                      "We hope you don't hate it."])

for result in results:
    print(result)

