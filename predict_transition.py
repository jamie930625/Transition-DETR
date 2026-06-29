import sys
sys.path.append('./model')

import torch
import os
import glob
from cue_detr_model import CuePointDetr
from cue_detr_data import CueTrainDataset
from transformers import DetrImageProcessor

def run_real_pipeline_demo():
    print("\n" + "="*65)
    print(" 🎧 DeepMIR DJ Transition: 生成式接歌挖空區間預測 (原曲對齊版) 🎧 ")
    print("="*65)

    ckpt_path = "checkpoints/exp_afternoon_demo_0/last.ckpt"
    model = CuePointDetr.load_from_checkpoint(ckpt_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    processor = DetrImageProcessor(size={'longest_edge': 355, 'shortest_edge': 128}, do_resize=False, do_pad=False)

    pair_prefix = "mix0000-00"
    track_a_name = f"{pair_prefix}_prev.png"
    track_b_name = f"{pair_prefix}_next.png"
    
    # 🚀 Demo 保命模式：直接帶入你剛剛提供的 CSV 真實數據，略過讀取檔案！
    gt_out_prev = 319.11       # Track A 原曲真實退場點
    gt_in_next = 7.84          # Track B 原曲真實進場點
    track_id_prev = "MS4fnBfTkf0"
    track_id_next = "6gGr8O40OXs"
    
    # 計算 60 秒擷取檔在原曲中的真實起點 (中心點往前推 30 秒)
    slice_start_prev = gt_out_prev - 30.0
    slice_start_next = gt_in_next - 30.0

    test_imgs = [track_a_name, track_b_name]
    dataset = CueTrainDataset(test_imgs, 'train_images/', processor)
    dataset.set_window_width(355)
    
    print(f"\n[*] 載入接歌任務: {pair_prefix}")
    print("="*24 + " 模型分析結果 " + "="*24)

    predictions_60s = {}
    predictions_abs = {}
    PX_TO_SEC = 512 / 22050

    for idx in range(len(dataset)):
        try:
            img_name = test_imgs[idx]
            is_prev = "prev" in img_name
            role = "Cue Out (退場點)" if is_prev else "Cue In (進場點)"
            
            pixel_values, _ = dataset[idx]
            img_tensor = pixel_values.unsqueeze(0).to(device)
            left_offset_px = dataset.last_slice[0]
            
            with torch.no_grad():
                outputs = model.model(pixel_values=img_tensor)
            
            logits = outputs.logits.squeeze() 
            boxes = outputs.pred_boxes.squeeze() 
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores, _ = probs[:, :-1].max(-1)
            best_idx = scores.argmax()
            
            pred_relative_px = boxes[best_idx].cpu().numpy()[0] * 355
            absolute_px_in_60s = left_offset_px + pred_relative_px
            pred_sec_in_60s = absolute_px_in_60s * PX_TO_SEC
            
            # 轉換為原曲絕對時間
            base_start_time = slice_start_prev if is_prev else slice_start_next
            abs_track_time = base_start_time + pred_sec_in_60s
            abs_track_time = max(0.0, abs_track_time) # 防呆
            
            predictions_60s[role] = pred_sec_in_60s
            predictions_abs[role] = abs_track_time
            
            # 格式化為 分:秒
            mins = int(abs_track_time // 60)
            secs = abs_track_time % 60
            
            print(f"📍 {img_name} {role}:")
            print(f"   -> 原曲絕對時間: 第 {abs_track_time:.2f} 秒 ({mins:02d}:{secs:05.2f})")
            print(f"   -> (參考) 60秒片段內第 {pred_sec_in_60s:.2f} 秒 (信心度: {scores[best_idx].item():.1%})")
            
        except Exception as e:
            print(f"讀取 {test_imgs[idx]} 失敗: {e}")

    print("\n" + "="*16 + " 給生成式團隊的 Action Item " + "="*16)
    if "Cue Out (退場點)" in predictions_abs and "Cue In (進場點)" in predictions_abs:
        out_time = predictions_abs['Cue Out (退場點)']
        in_time = predictions_abs['Cue In (進場點)']
        
        print(f"✅ 模型建議 (可直接對應原曲 MP3)：")
        print(f"🎵 Track A 原曲 (ID: {track_id_prev}):")
        print(f"   請在原曲的第 {out_time:.2f} 秒 ({int(out_time//60):02d}:{out_time%60:05.2f}) 處開始 Fade Out / 挖空。")
        print(f"🎵 Track B 原曲 (ID: {track_id_next}):")
        print(f"   請在原曲的第 {in_time:.2f} 秒 ({int(in_time//60):02d}:{in_time%60:05.2f}) 處開始 Fade In / 銜接。")
        print(f"👉 請依據上述秒數裁切原音檔，並交由 Generative Model 生成過渡橋段！")
    print("=================================================================\n")

if __name__ == '__main__':
    run_real_pipeline_demo()
