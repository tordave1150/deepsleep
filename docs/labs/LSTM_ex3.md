# Lab LSTM_ex3

[Lab Index](README.md) · [Lecture 7](../lectures/07-lstm-gru.md) · [Index](../INDEX.md)

## เป้าหมาย

ฝึก BiLSTM แยก sentiment ของ IMDB แล้วตามผลของ padding, pooling, LayerNorm และ regularization

ต้นฉบับ [class07/code/LSTM_ex3.ipynb](https://drive.google.com/file/d/1K4EYGsffOxa9EBnWlGqAPHdpdHi2Xx9c/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, Keras IMDB, NumPy, Matplotlib, PyTorch และ torchinfo ดาวน์โหลดข้อมูลผ่าน Keras ถ้าไม่มี cached copy

## Token IDs, Lengths และ Padding

อ่าน Keras IMDB ที่ tokenize แล้ว จำกัด vocabulary 10,000 และ pad/truncate ที่ 300 ตัว Token ID เป็น long มี label 0 หรือ 1 Dataset เก็บ lengths จากจำนวน token ที่ไม่ใช่ 0 และส่ง length แยกกับข้อมูลเพื่อใช้ packing ตอนฝึกใช้ post-padding แบ่งชุด train เดิมเป็น train/validation 80:20 และเก็บ test แยก

> **อ้างอิง** [class07/code/LSTM_ex3.ipynb](https://drive.google.com/file/d/1K4EYGsffOxa9EBnWlGqAPHdpdHi2Xx9c/view) · cells 11–21, 28–29

## BiLSTM และการเลือก Feature

Embedding เปลี่ยน (B,T) เป็น (B,T,128) แล้ว BiLSTM สองชั้นให้ (B,T,2H) โหมด pack เอา Padding ออกจาก recurrence และ unpack ด้วยค่า −1e9 เพื่อให้ max ไม่เลือกช่อง padding Last pooling จับคู่ row กับ length−1 ทาง forward และหยิบเวลา 0 ทาง backward แล้ว concat เป็น (B,2H) ใช้ LayerNorm และ Dropout ตาม config ก่อน Sigmoid ให้ (B,1)

> **อ้างอิง** [class07/code/LSTM_ex3.ipynb](https://drive.google.com/file/d/1K4EYGsffOxa9EBnWlGqAPHdpdHi2Xx9c/view) · cells 22–23

## ฝึกสี่ Config และเปรียบเทียบ

ตั้ง baseline แบบ max pooling แล้วเปลี่ยนเป็น packing/last pooling/LayerNorm ในชุดเดียวกัน จากนั้นเพิ่ม Dropout 0.3 และแบบสุดท้ายลด hidden size พร้อมเพิ่ม Weight Decay เนื่องจากมีการเปลี่ยนหลายปัจจัยพร้อมกัน จึงตีความผลได้ในระดับ config ก่อนจะสรุปผลของตัวแปรใดตัวแปรหนึ่ง

> **อ้างอิง** [class07/code/LSTM_ex3.ipynb](https://drive.google.com/file/d/1K4EYGsffOxa9EBnWlGqAPHdpdHi2Xx9c/view) · cells 24–36

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 23

```python
last_fwd = out[torch.arange(batch_size), lengths - 1, :self.hidden_dim]
last_bwd = out[:, 0, self.hidden_dim:]
```

forward เลือกตำแหน่งตาม length รายแถว ส่วน backward อ่าน feature ครึ่งหลังที่เวลาแรก

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 32 มี Learning Curves และ cells 34–35 รายงานเวลาฝึกและ test metrics ของสี่แบบ


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ลองวาดตำแหน่ง last forward/backward ของ review ที่มี padding สองช่อง ถ้า tensor indexing ให้ (B,B,H) ให้ตรวจว่าจับคู่ row กับ length รายตัวอย่างหรือใช้ดัชนีเดียวกับทุกแถว

[Lab Index](README.md) · [Lecture 7](../lectures/07-lstm-gru.md)
