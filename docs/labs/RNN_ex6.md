# Lab RNN_ex6

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md) · [Index](../INDEX.md)

## เป้าหมาย

ตาม Seq2Seq ที่ใช้ RNN Encoder–Decoder สำหรับหลาย horizon และแยก Teacher Forcing จาก roll-out

ต้นฉบับ [class06/code/RNN_ex6.ipynb](https://drive.google.com/file/d/1BShS_DdKwMCOTBwabcUP112Jkqsh2S0U/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, Matplotlib, PyTorch และ torchinfo สัญญาณสร้างใน Notebook

## เตรียมข้อมูลและ Decoder Context

ใช้ noisy sine แบ่งตามเวลา X=(B,50,1) และ Y=(B,N,1) Encoder คืน h_n ขนาด (1,B,128) เป็น state เริ่มต้นของ Decoder ส่วน Decoder Input ครั้งแรกคือค่าจริงสุดท้ายของ Window มี shape (B,1,1)

> **อ้างอิง** [class06/code/RNN_ex6.ipynb](https://drive.google.com/file/d/1BShS_DdKwMCOTBwabcUP112Jkqsh2S0U/view) · cells 10–19

## Teacher Forcing และเส้นทาง Inference

Decoder สร้าง pred ทีละค่าและส่ง hidden state ต่อ ถ้ามี y_target จะใช้ค่าจริงที่ timestep ปัจจุบันเป็น input สำหรับครั้งถัดไป หากไม่มี target จะใช้ pred เอง สุดท้าย cat ตามเวลาได้ (B,N,1) Experiments เทียบ N=3,14,25 ตาม cell 23

> **อ้างอิง** [class06/code/RNN_ex6.ipynb](https://drive.google.com/file/d/1BShS_DdKwMCOTBwabcUP112Jkqsh2S0U/view) · cells 17–25

## อ่านกราฟและข้อจำกัด

กราฟเปรียบเทียบ Learning Curves และ Test Prediction สำหรับ horizons ต่างกัน ดูความไวต่อ initial weights และผลของ N_FUTURE ตาม discussion การประเมินที่เปิด y_target ให้ Decoder กับการคาดจริงที่ไม่มี target ต้องอ่านแยกกัน

> **อ้างอิง** [class06/code/RNN_ex6.ipynb](https://drive.google.com/file/d/1BShS_DdKwMCOTBwabcUP112Jkqsh2S0U/view) · cells 26–33

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 19

```python
if y_target is not None:
    curr_input = y_target[:, t, :].unsqueeze(1)
else:
    curr_input = pred
```

ทั้งสองแขนงส่ง Tensor รูป (B,1,1) ไปยัง Decoder เดียวกัน

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 28 และ 30 เก็บรูป Training History และ Test Prediction ในกราฟ cell 30 แบบ horizon 3 เสียรูปคลื่นหลังการป้อนผลกลับหลายครั้ง ขณะที่ horizon 25 ตามคาบได้ใกล้กว่าในช่วงที่แสดง Cell 33 อภิปรายความต่างนี้เฉพาะกับการตั้งค่าใน Lab


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ทำแผนภาพ input ของ Decoder สาม step เมื่อมี target และเมื่อมีเพียง prediction หาก cat ผิดแกน ให้ตรวจว่าจำนวน Batch คงเดิมและเพิ่มเฉพาะแกนเวลา

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md)
