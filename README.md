
# Transition DETR

這是一個自動預測 DJ 混音 Cue In / Cue Out 接歌點的模型。

## 1. 環境設定

```bash
pip install -r requirements.txt

```

## 2. 下載模型權重

請至以下連結下載模型權重 last.ckpt：
https://drive.google.com/file/d/17DKCkQ9pK9i2O29O4VpIQgwyuz4R0dkA/view?usp=share_link

下載後，請在專案根目錄建立對應資料夾，並將檔案放入，確保路徑結構如下：
```text
checkpoints/exp_afternoon_demo_0/last.ckpt

```

## 3. 執行預測

此腳本為 Demo 模式，會直接讀取 `train_images/` 內的預設圖片 (`mix0000-00`) 進行預測。

請在終端機執行以下指令：

```bash
python predict_transition.py

```

## 4. 預期輸出

執行成功後，終端機會印出分析結果，可用於後續的 Generative Model，包含：

* **Track A** 建議的 cue Out 絕對時間 (秒)
* **Track B** 建議的 cue In 絕對時間 (秒)

```

```
