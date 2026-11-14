# Lab MLP1

[Lab Index](README.md) · [Lecture 2](../lectures/02-mlp-basics.md) · [Index](../INDEX.md)

## เป้าหมาย

ไล่การฝึก MLP แบบ single-label multiclass ตั้งแต่สร้างข้อมูลจำลองจนถึง Inference พร้อมอ่านความต่างของ train, validation และ test

ต้นฉบับ [class02/code/MLP1.ipynb](https://drive.google.com/file/d/1F-4hVaszxs0o6ZT2P4xjn_rcINQODFe9/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, random, time, NumPy, Matplotlib, torchinfo และ PyTorch Cells 3–5 ตรวจ NVIDIA, ติดตั้ง torchinfo และ import Cell 8 เลือก CUDA → MPS → CPU ให้ model, input และ target ที่ส่งเข้า Loss อยู่บน device เดียวกัน

## ข้อมูลและชนิด Tensor

ตั้ง seed ก่อนสร้างข้อมูล 100 features ต่อแถวและสุ่ม label จำนวนเต็ม 0–9 แยกจาก features ชุด train/validation มี 10,000 แถว แบ่ง 8,000/2,000 และสร้าง test อีก 2,000 แถว แปลง features เป็น float32 และ labels เป็น long เพื่อให้เข้ากับ Linear และ CrossEntropyLoss ตามลำดับ DataLoader ใช้ batch size 64 และสลับลำดับเฉพาะชุดฝึก

> **อ้างอิง** [class02/code/MLP1.ipynb](https://drive.google.com/file/d/1F-4hVaszxs0o6ZT2P4xjn_rcINQODFe9/view) · cells 6–13

## โครงข่ายและจำนวนพารามิเตอร์

Sequential ส่ง (B,100) → (B,256) → (B,128) → (B,64) → (B,10) ReLU อยู่หลัง Hidden Linear แต่ละชั้น Weight ชั้นแรกมี shape (256,100) และ Bias มี 256 ค่า นับแบบเดียวกันทุก Linear แล้วรวมได้ 67,658 ตาม summary ค่า 10 ตำแหน่งในแต่ละแถวคือ logits

> **อ้างอิง** [class02/code/MLP1.ipynb](https://drive.google.com/file/d/1F-4hVaszxs0o6ZT2P4xjn_rcINQODFe9/view) · cells 14–21

## Training, Validation และ Learning Rate

ตั้ง CrossEntropyLoss, AdamW ที่ LR 0.001 และ CosineAnnealingLR T_max=10 ในหนึ่ง Batch ล้าง Gradient เดิม คำนวณ logits และ Loss ทำ backward แล้วอัปเดต Weight ผล Loss เป็นค่าเฉลี่ยของ Batch จึงคูณด้วยจำนวนตัวอย่างก่อนสะสมและหารด้วยจำนวนทั้งหมดตอนจบ Epoch

Validation ใช้ eval() และ inference_mode() โดยเก็บ Metric โดยไม่ทำ backward จากนั้นเรียก scheduler.step() หนึ่งครั้งต่อ Epoch และบันทึก train/val loss กับ accuracy ลง history Cell 27 วาดกราฟสองแกน

> **อ้างอิง** [class02/code/MLP1.ipynb](https://drive.google.com/file/d/1F-4hVaszxs0o6ZT2P4xjn_rcINQODFe9/view) · cells 22–27

## Test และ Inference

Cell 29 ใช้ test tensor ทั้งชุดคำนวณ Accuracy จาก argmax ของ logits เทียบ label จริง Cell 31 เลือกหนึ่งแถว shape (100,) แล้วใช้ unsqueeze(0) ให้เป็น (1,100) เพื่อให้ argmax(dim=1) ชี้แกนคลาส Cell 32 ใช้ slice ห้าแถวจึงมี Batch axis อยู่แล้ว

> **อ้างอิง** [class02/code/MLP1.ipynb](https://drive.google.com/file/d/1F-4hVaszxs0o6ZT2P4xjn_rcINQODFe9/view) · cells 28–32

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 26

```python
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Gradient ของ criterion ไหลจาก logits ไปยัง Weight ของชั้นที่ใช้สร้าง logits แล้ว Optimizer อัปเดต Weight

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 26 บันทึกว่า train accuracy ขึ้นจาก 9.70% เป็น 60.30% และ validation accuracy ใน Epoch สุดท้ายอยู่ที่ 9.75% Cell 27 แสดง train loss ลงและ val loss ขึ้น Cell 29 บันทึก test accuracy 9.35% เมื่อ features กับ labels ถูกสุ่มแยกกัน ตัวอย่างนี้สาธิตการฝึกและการจำข้อมูลฝึกโดยไม่มีหลักฐานว่าเรียนความสัมพันธ์ที่ใช้กับชุดทดสอบ


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** คำนวณ shape ของ logits และ labels เมื่อ Batch มี 16 ตัวอย่าง แล้วคำนวณค่าเฉลี่ย Loss จาก Batch ขนาด 64 กับ 16 แบบถ่วงจำนวนตัวอย่าง หาก Loss ฟ้องชนิดข้อมูล ให้ตรวจว่า target เป็น class index แบบ long จาก cell 13

[Lab Index](README.md) · [Lecture 2](../lectures/02-mlp-basics.md)
