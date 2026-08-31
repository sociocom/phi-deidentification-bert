# 医療テキスト仮名化 Modern-BERT版
医療テキスト仮名化を行う[PHI-Deidentification](https://github.com/sociocom/PHI-Deidentification)のベースモデルをModern-BERTにしたものです。

## Install
```
uv sync
```

## 学習

`configs/sample.yaml`をコピーしてconfigを作成。

```
python train.py configs/sample.yaml
```

## 推論

```
python inference.py configs/sample.yaml
```
