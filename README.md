# 如何使用？
## 資料集
- 資料集放置在 `./datasets` 內，採CSV格式。
- 我們使用 `UNB` 與 [Ginst Engelend](https://github.com/GintsEngelen/WTMC2021-Code) 版本的 `CICIDS2017` 資料集作為訓練集，分別用於`M1`模型與`M2`模型，他們本質上是不同的，儘管原始PCAP相同。
- 在本研究中，我們使用 `UNB` 的 `CICIDS2019` 資料集作為未知資料集。
## 訓練模型
- 執行 `train.py -m [model]`，`-m` 參數可以是`1`或`2`，利用`-h`可以看更多參數。
- `-s`參數可以試著替換`seed`
## 測試未知性能
- 執行 `eval.py -m 1`，`-m` 參數可以是`1`或`2`，利用`-h`可以看更多參數。
- `-s`參數可以試著替換`seed`
- `eval.py`會測試未知資料集
- `M1`與`M2`資料夾可以找到SSE的分布輸出與混淆矩陣
- `logs`內會有各檔案、各模型的性能數據，檔案名稱是 `M{1}_sd_{seed}`，`filter`表示當前過濾器過濾後測試集僅有哪個類別的數據。
## 增量訓練
- `incrementTrain.ipynb`內為產出增量訓練後結果為主，初始新增分類權重與偏差為隨機、新增類的SLCPL中心為0。
- 繪製的圖在 `M{1}` 裡面，`noOD`後綴為沒增量訓練，`Bin`則為彙整成良性與惡意的情境。

# How to use?
## the dataset
- Put datasets in `./datasets`, datasets are `csv` formats.
- We used the `CICIDS2017` datasets from `UNB` for the `M1` model and [Ginst Engelend](https://github.com/GintsEngelen/WTMC2021-Code) editions for the `M2` model(They are different although their source of PCAP is the same.).
- We used the `CICIDS2019` dataset from `UNB` as the unknown dataset in this work.
## Train
- Run `train.py -m [model]` model allow `1` for the `M1` or `2` for the `M2`, and use `-h` to show other arguments.
- `-s` arguments can assign the `randomseed`
## Eval for unknown
- Run `eval.py -m [model]` model allow `1` or `2`, and use `-h` to show other arguments.
- `-s` arguments can choose the `randomseed` after training with specified seeds.
- `eval.py` will test unknown datasets.
- You can find the SSE output and confusion matrix in folders `M1` and `M2`.
- In the folder named `logs`, we will generate some files with the performance of each model and seed in close-set and open-set datasets.
- The filename format is `M{1}_sd_{seed}`, `filter` field in log data means which class of data is in the test dataset of this record.
## Incremential Train
- `incrementTrain.ipynb` is a file for the incremental train. This step will create a new class of `unknown` with random bias and weight. In SLCPL centers, this class has a zero center.
- The model will put some images in the `M[model] folder. If `noOD` in the postfix of the image name means without the incremental train, `Bin` means summarised multiclass to binary of benign and malicious.