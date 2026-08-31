import sys

from omegaconf import OmegaConf
import torch
from transformers import pipeline

num_table = str.maketrans(
    "０１２３４５６７８９",
    "0123456789",
)


def run():
    cfg_path = sys.argv[1]
    cfg = OmegaConf.load(cfg_path)

    model_dir = cfg.inference.model_dir
    ner = pipeline(
        task="token-classification",
        model=model_dir,
        tokenizer=model_dir,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=cfg.inference.batch_size,
    )

    with open(cfg.inference.data_path, encoding="utf-8") as f:
        texts = [line.strip() for line in f]
        texts = [t.translate(num_table) for t in texts]
    predictions = ner(texts)

    tagged_texts = []
    for text, entities in zip(texts, predictions):
        tagged_text = text

        for entity in reversed(entities):
            start = entity["start"]
            end = entity["end"]
            label = entity["entity_group"]

            tagged_text = (
                    tagged_text[:start]
                    + f"<{label}>"
                    + tagged_text[start:end]
                    + f"</{label}>"
                    + tagged_text[end:]
            )

        tagged_texts.append(tagged_text)


if __name__ == '__main__':
    run()
