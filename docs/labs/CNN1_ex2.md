# Lab CNN1_ex2

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md) · [Index](../INDEX.md)

## เป้าหมาย

สร้าง CNN สายตรงสองแบบและตรวจว่า shape กับจำนวนพารามิเตอร์ตรงกัน

ต้นฉบับ [class04/code/CNN1_ex2.ipynb](https://drive.google.com/file/d/1tqjD0G5uMVWwPUiQba_F5JIbTRqiOHRh/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, PyTorch, torchinfo, torchview และ torchviz การวาดกราฟอาศัยตัวเรนเดอร์ Graphviz ส่วน timing ใช้ %%timeit ใน Notebook

## Input และ Feature Extractor

เลือกภาพจำลอง (1,502,502) หรือ (3,502,502) แล้วเพิ่ม Batch axis สำหรับ Conv2d ขนาด 7 และ 5 แบบ valid ลดด้านภาพ 502→496→492 แล้ว MaxPool ให้ 246 Conv ขนาด 3 แบบ padding 1 รักษาขนาด 246 ก่อนที่ Conv สองชั้น stride 2 ลดเป็น 123 และ 61 ตามลำดับ ผลสุดท้ายมี 16 channels ดังนั้น Flatten ได้ 16×61×61 = 59,536 Features ต่อภาพ ใช้ dummy input หนึ่งภาพตรวจขนาดเดียวกันก่อนสร้าง Linear

> **อ้างอิง** [class04/code/CNN1_ex2.ipynb](https://drive.google.com/file/d/1tqjD0G5uMVWwPUiQba_F5JIbTRqiOHRh/view) · cells 10–16

## Sequential เทียบ Subclassing

แบบแรกวาง extractor และ classifier ใน Sequential ส่วนแบบที่สองประกาศสมาชิกใน MySequentialCNN และเรียกตามลำดับใน forward สองแบบใช้หัว 1024→1024→1000 เมื่อสร้างโมเดลคนละครั้ง น้ำหนักเริ่มต้นเป็นคนละชุดแม้ topology เหมือนกัน

> **อ้างอิง** [class04/code/CNN1_ex2.ipynb](https://drive.google.com/file/d/1tqjD0G5uMVWwPUiQba_F5JIbTRqiOHRh/view) · cells 17–29

## ทดลอง Forward

สร้าง random input batch 7 ภาพ ผลที่คาดจากโค้ดคือ logits (7,1000) %%timeit วัดเฉพาะทางผ่านโมเดล ตัวอย่างใช้ข้อมูลสุ่มและไม่มี label, Loss หรือการฝึก

> **อ้างอิง** [class04/code/CNN1_ex2.ipynb](https://drive.google.com/file/d/1tqjD0G5uMVWwPUiQba_F5JIbTRqiOHRh/view) · cells 30–33

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 15

```python
dummy_output = model(dummy_input.unsqueeze(0))
```

unsqueeze เพิ่ม Batch ก่อน Forward เพื่อหา Flatten size จากสถาปัตยกรรมจริง

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 20 และ 27 รายงาน 63,289,864 พารามิเตอร์ทั้งสองแบบ และ cells 21–22, 28–29 เก็บกราฟ SVG Cells 32–33 เก็บเวลาเฉพาะ session ต้นฉบับ


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เปลี่ยน input จาก 1 เป็น 3 channels แล้วระบุว่า shape ของ Weight ชั้นแรกเปลี่ยนอย่างไรและผลลัพธ์ยังมี 1,000 คอลัมน์เพราะอะไร หาก Linear shape ไม่ตรง ให้ตรวจ dummy pass กับ shape ที่ใช้จริง

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md)
