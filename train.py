import re
import sys

import numpy as np
from datasets import Dataset
from omegaconf import OmegaConf
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

type Tokenizer = PreTrainedTokenizerBase

num_table = str.maketrans(
    "０１２３４５６７８９",
    "0123456789",
)


def run():
    cfg_path = sys.argv[1]
    cfg = OmegaConf.load(cfg_path)

    model_name = cfg.model.name
    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = cfg.model.max_length

    labels = ["O"] + [l for label in cfg.labels for l in [f"B-{label}", f"I-{label}"]]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # データ
    with open(cfg.train.data_path, encoding="utf-8") as f:
        train_texts = [line.strip() for line in f]
    train_texts, eval_texts = train_test_split(
        train_texts,
        test_size=0.1,
        random_state=cfg.train.seed,
        shuffle=True,
    )
    # train_texts = all_texts
    # eval_texts = None
    train_dataset = build_dataset(tokenizer, train_texts, label2id)
    eval_dataset = (
        build_dataset(tokenizer, eval_texts, label2id)
        if eval_texts is not None
        else None
    )

    output_dir = cfg.train.output_dir
    training_args = TrainingArguments(
        # 学習設定
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        bf16=True,
        # 評価・保存
        output_dir=output_dir,
        eval_strategy=cfg.train.eval_strategy,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=cfg.train.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )
    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print("評価結果:", metrics)

        export_eval_errors(
            trainer=trainer,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            id2label=id2label,
        )

    # 最良モデルとトークナイザーを保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def build_dataset(
    tokenizer: Tokenizer, tagged_texts: list[str], label2id: dict[str, int]
):
    records = []

    for tagged_text in tagged_texts:
        plain_text, entities = parse_tagged_text(tagged_text)

        encoded = tokenizer(
            plain_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
        )

        label_ids = []
        for start, end in encoded["offset_mapping"]:
            # [CLS], [SEP]など
            if start == end:
                label_ids.append(-100)
                continue

            tag = "O"
            for ent_start, ent_end, ent_label in entities:
                # tokenとentity範囲が重なっているか
                if start < ent_end and end > ent_start:
                    if start == ent_start:
                        tag = f"B-{ent_label}"
                    else:
                        tag = f"I-{ent_label}"
                    break
            label_ids.append(label2id[tag])

        records.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": label_ids,
            }
        )

    return Dataset.from_list(records)


def parse_tagged_text(tagged_text: str):
    """
    タグ付きテキストから
    - タグを除去したplain_text
    - エンティティ範囲: (start, end, label)
    を返す
    """
    plain_parts = []
    entities = []
    cursor = 0
    plain_pos = 0

    phi_pattern = re.compile(r"<(phi_[a-zA-Z0-9_]+)>(.*?)</\1>", re.DOTALL)
    for m in phi_pattern.finditer(tagged_text):
        # タグ前の通常テキスト
        before = tagged_text[cursor : m.start()]
        plain_parts.append(before)
        plain_pos += len(before)

        label = m.group(1)  # phi_location
        value = m.group(2)  # 白波台

        start = plain_pos
        end = plain_pos + len(value)

        plain_parts.append(value)
        entities.append((start, end, label))

        plain_pos = end
        cursor = m.end()

    # 最後の通常テキスト
    rest = tagged_text[cursor:]
    plain_parts.append(rest)

    return "".join(plain_parts), entities


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # 各トークンで最もスコアが高いラベルIDを選択
        predicted_ids = np.argmax(predictions, axis=-1)

        true_predictions = []
        true_labels = []

        for predicted_sequence, label_sequence in zip(
            predicted_ids,
            labels,
        ):
            sequence_predictions = []
            sequence_labels = []

            for predicted_id, label_id in zip(
                predicted_sequence,
                label_sequence,
            ):
                # 特殊トークンやパディングを評価から除外
                if label_id == -100:
                    continue

                sequence_predictions.append(id2label[int(predicted_id)])
                sequence_labels.append(id2label[int(label_id)])

            true_predictions.append(sequence_predictions)
            true_labels.append(sequence_labels)

        return {
            "precision": precision_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
        }

    return compute_metrics


def export_eval_errors(
    trainer: Trainer,
    eval_dataset,
    tokenizer: Tokenizer,
    id2label: dict[int, str],
    max_error_sentences: int | None = None,
):
    """
    eval_datasetに対して予測し、トークン単位の誤りを含む文だけを標準出力する。

    trainer.train()後に呼び出すと、
    load_best_model_at_end=Trueの場合は最良モデルで評価される。

    max_error_sentences:
        表示する誤り文の最大数。
        Noneの場合はすべて表示する。
    """
    prediction_output = trainer.predict(
        eval_dataset,
        metric_key_prefix="eval",
    )

    predictions = prediction_output.predictions
    label_ids = prediction_output.label_ids

    # モデルによってpredictionsがtupleになる場合への対応
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1)

    num_error_sentences = 0
    num_error_tokens = 0
    num_evaluated_tokens = 0

    for sample_index, sample in enumerate(eval_dataset):
        input_ids = sample["input_ids"]
        sequence_length = len(input_ids)

        sample_gold_ids = label_ids[sample_index][:sequence_length]
        sample_pred_ids = pred_ids[sample_index][:sequence_length]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_details = []
        has_error = False

        for token_index, (token, gold_id, pred_id) in enumerate(
            zip(tokens, sample_gold_ids, sample_pred_ids)
        ):
            gold_id = int(gold_id)
            pred_id = int(pred_id)

            # special token、padding、評価対象外サブワードなど
            if gold_id == -100:
                continue

            gold_label = id2label[gold_id]
            pred_label = id2label[pred_id]
            is_error = gold_label != pred_label

            num_evaluated_tokens += 1

            if is_error:
                has_error = True
                num_error_tokens += 1

            token_details.append(
                {
                    "token_index": token_index,
                    "token": token,
                    "gold": gold_label,
                    "pred": pred_label,
                    "error": is_error,
                }
            )

        # 完全に正解した文は表示しない
        if not has_error:
            continue

        num_error_sentences += 1

        # 表示数に上限を設ける場合
        should_print = (
            max_error_sentences is None or num_error_sentences <= max_error_sentences
        )

        if should_print:
            decoded_text = tokenizer.decode(
                input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            print()
            print("=" * 100)
            print(f"sample_index: {sample_index}")
            print(f"text: {decoded_text}")
            print("-" * 100)
            print(f"{'TOKEN':<25}{'GOLD':<30}{'PRED':<30}{'RESULT'}")
            print("-" * 100)

            for detail in token_details:
                marker = "<<< ERROR" if detail["error"] else ""

                print(
                    f"{detail['token']:<25}"
                    f"{detail['gold']:<30}"
                    f"{detail['pred']:<30}"
                    f"{marker}"
                )

    summary = {
        "num_eval_sentences": len(eval_dataset),
        "num_error_sentences": num_error_sentences,
        "sentence_error_rate": (
            num_error_sentences / len(eval_dataset) if len(eval_dataset) > 0 else 0.0
        ),
        "num_evaluated_tokens": num_evaluated_tokens,
        "num_error_tokens": num_error_tokens,
        "token_error_rate": (
            num_error_tokens / num_evaluated_tokens if num_evaluated_tokens > 0 else 0.0
        ),
        "metrics": prediction_output.metrics,
    }

    print()
    print("=" * 100)
    print("評価エラー集計")
    print("-" * 100)
    print(f"評価文数       : {summary['num_eval_sentences']}")
    print(f"誤りを含む文数 : {summary['num_error_sentences']}")
    print(f"文単位誤り率   : {summary['sentence_error_rate']:.4%}")
    print(f"評価トークン数 : {summary['num_evaluated_tokens']}")
    print(f"誤りトークン数 : {summary['num_error_tokens']}")
    print(f"トークン誤り率 : {summary['token_error_rate']:.4%}")
    print(f"評価指標       : {summary['metrics']}")

    if max_error_sentences is not None and num_error_sentences > max_error_sentences:
        omitted = num_error_sentences - max_error_sentences
        print(
            f"※ 誤り文は先頭{max_error_sentences}件のみ表示しました。"
            f"残り{omitted}件は省略しています。"
        )

    return summary


if __name__ == "__main__":
    run()
