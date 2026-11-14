# Lab Attention_ex1

[Lab Index](README.md) · [Lecture 8](../lectures/08-attention-transformer.md) · [Index](../INDEX.md)

## เป้าหมาย

เทียบ LSTM, GRU และ Transformer Encoder ที่เรียนจากศูนย์บน IMDB ในสองขนาดโมเดล

ต้นฉบับ [class08/code/Attention_ex1.ipynb](https://drive.google.com/file/d/1qSxbu91Aj63DYCA2BFMTiVLcOvMVF_PR/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, Keras IMDB, NumPy, Matplotlib, PyTorch และ torchinfo โค้ดวัดเวลาและหน่วยความจำในสภาพแวดล้อมที่รองรับ

## การเตรียมข้อมูลร่วม

โหลด Keras IMDB ที่ tokenize แล้ว ตัด vocabulary ที่ 10,000 และใช้ความยาว 300 ส่ง token IDs (B,T), label และ lengths ไปยังโมเดล แบ่ง 80% ของชุด train เดิมเป็น train และที่เหลือเป็น validation ส่วน test เก็บแยก

> **อ้างอิง** [class08/code/Attention_ex1.ipynb](https://drive.google.com/file/d/1qSxbu91Aj63DYCA2BFMTiVLcOvMVF_PR/view) · cells 11–22

## สองเส้นทางจาก Tokens สู่ Binary Output

LSTM/GRU ใช้ recurrent encoder และอ่าน state ของลำดับจริง Transformer ใช้ token embedding บวก learnable positional embedding แล้วผ่าน encoder สองชั้นสี่ heads ใช้ x == 0 เป็น key padding mask จากนั้นลบตำแหน่ง padding ออกจาก masked mean ก่อน Linear และ Sigmoid ผลทั้งสามแบบเป็น (B,1)

> **อ้างอิง** [class08/code/Attention_ex1.ipynb](https://drive.google.com/file/d/1qSxbu91Aj63DYCA2BFMTiVLcOvMVF_PR/view) · cells 23–27

## หก Config และการประเมิน

Standard ใช้ embedding 64 และ recurrent hidden 64 ส่วน Starving ใช้ embedding 16 และ hidden 32 ฝึกทั้งสามสถาปัตยกรรมในแต่ละ tier ด้วยข้อมูลเดียวกันและใช้ AdamW กับ BCELoss สำหรับผลจาก Sigmoid ใช้ผลกราฟดู loss, accuracy, test, time และ VRAM การเปลี่ยน tier ทำให้ต้องเทียบจำนวนพารามิเตอร์ของแต่ละ architecture ประกอบ

> **อ้างอิง** [class08/code/Attention_ex1.ipynb](https://drive.google.com/file/d/1qSxbu91Aj63DYCA2BFMTiVLcOvMVF_PR/view) · cells 28–43

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 27

```python
padding_mask = (x == 0)
```

mask มี shape (B,T) และตาม convention นี้ True คือ PAD ที่ต้องกันออก

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 37 และ 39 เก็บกราฟการฝึก และ cell 41 เทียบ test accuracy กับทรัพยากร ให้อ่านตัวเลขประกอบกับขนาดและการตั้งค่าของ run


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** สำหรับ Tokens จริง 3 ตำแหน่งกับ PAD 2 ตำแหน่ง ให้ระบุ mask ของ Key และตัวหารของ masked mean ข้อสังเกตจากโค้ดคือ input ที่มี PAD ล้วนทำให้ตัวหารของ Transformer เป็นศูนย์ จึงต้องตรวจข้อมูลก่อนใช้กับ Dataset อื่น

[Lab Index](README.md) · [Lecture 8](../lectures/08-attention-transformer.md)
