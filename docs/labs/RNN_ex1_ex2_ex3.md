# Lab RNN_ex1_ex2_ex3

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md) · [Index](../INDEX.md)

## เป้าหมาย

เปรียบเทียบ Window และจำนวน RNN Layers สำหรับ next-step แล้วสังเกตการสะสม error เมื่อทำ recursive forecast

ต้นฉบับ [class06/code/RNN_ex1_ex2_ex3.ipynb](https://drive.google.com/file/d/1fa1r5p110siHgvGrksP_dsgQP90m4sUk/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, Matplotlib, PyTorch และ torchinfo สร้างข้อมูลจำลองใน Notebook และเลือก CUDA/MPS/CPU

## สร้างสัญญาณและ Window

สร้าง 5,000 จุดจาก sine บวก Gaussian Noise 0.10 แบ่งเวลา 70% train, 15% validation และ 15% test ก่อนทำ Windows SineDataset เก็บข้อมูลเป็นคอลัมน์และคืน Window (W,1) คู่กับค่าจริงถัดไป (1,)

> **อ้างอิง** [class06/code/RNN_ex1_ex2_ex3.ipynb](https://drive.google.com/file/d/1fa1r5p110siHgvGrksP_dsgQP90m4sUk/view) · cells 9–15

## ฝึกสี่การตั้งค่า

SimpleRNNModel ใช้ batch_first, hidden size 64 และหัว Linear จาก timestep สุดท้ายให้ผล (B,1) เทียบ W=5 หรือ 50 กับ 1 หรือ 3 Layers ใช้ Adam, MSE และ 50 Epochs กราฟ learning curves ช่วยดูว่าความต่างเกิดระหว่างฝึกหรือ validation

> **อ้างอิง** [class06/code/RNN_ex1_ex2_ex3.ipynb](https://drive.google.com/file/d/1fa1r5p110siHgvGrksP_dsgQP90m4sUk/view) · cells 16–26

## Single-step และ Recursive

Cell 29 เลื่อน Window ด้วยค่าจริงใน test ทุกครั้ง ส่วน cell 31 เลื่อน Window ด้วย prediction จาก model p มี (1,1) จึง unsqueeze(1) เป็น (1,1,1) ก่อน cat ตามแกนเวลาเพื่อคง Window length Cell 33 แสดง forecast 400 จุดจาก seed Window จริง

> **อ้างอิง** [class06/code/RNN_ex1_ex2_ex3.ipynb](https://drive.google.com/file/d/1fa1r5p110siHgvGrksP_dsgQP90m4sUk/view) · cells 27–33

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 31

```python
current_window = torch.cat((current_window[:, 1:, :], p.unsqueeze(1)), dim=1)
```

ตัด timestep แรกทิ้งแล้วเติมหนึ่ง timestep ที่ท้าย Window จึงคงความยาวเดิม

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 26, 29 และ 33 เก็บกราฟการฝึก, one-step และ recursive ใช้รูปที่ 29 กับ 33 คู่กันเพราะการทำนายที่ใกล้จริงเมื่อมีบริบทจริงอาจเบี่ยงออกเมื่อบริบทมาจากโมเดลเอง


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ไล่ Window ขนาด 5 สองรอบเมื่อ prediction แรกสูงกว่าค่าจริง ถ้าขนาด Window โตขึ้นทุกรอบ ให้ตรวจ slice [:, 1:, :] และแกนที่ใช้ cat

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md)
