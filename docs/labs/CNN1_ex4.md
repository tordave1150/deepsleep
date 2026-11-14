# Lab CNN1_ex4

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md) · [Index](../INDEX.md)

## เป้าหมาย

เปรียบเทียบ MLP กับ CNN บน MNIST ด้วยชุดข้อมูลและวงจรฝึกเดียวกัน

ต้นฉบับ [class04/code/CNN1_ex4.ipynb](https://drive.google.com/file/d/190UnS1etlKEyqRYwjxHnP4_jGYKXVT53/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision, NumPy, Matplotlib และ torchinfo โค้ดดาวน์โหลด MNIST และเลือก CUDA, MPS หรือ CPU

## ดูข้อมูลและ Normalize

MNIST เป็นภาพเทาขนาด 28×28 และป้ายตัวเลข 0–9 แปลงเป็น Tensor แล้ว Normalize ด้วย mean 0.1307 และ std 0.3081 ตาม cell 14 แยก train 50,000 กับ validation 10,000 และคง test loader ไว้ประเมินปลายทาง

> **อ้างอิง** [class04/code/CNN1_ex4.ipynb](https://drive.google.com/file/d/190UnS1etlKEyqRYwjxHnP4_jGYKXVT53/view) · cells 9–16

## สถาปัตยกรรมสองแบบ

MLP เริ่มด้วย Flatten เป็น (B,784) แล้ว Linear 128→64→10 CNN เก็บรูปเชิงพื้นที่ขณะผ่าน Conv 16 channels และ 32 channels คั่น Pooling ทำให้ (B,1,28,28) → (B,16,14,14) → (B,32,7,7) แล้ว Flatten 1,568 ค่าเข้า Linear 64→10

> **อ้างอิง** [class04/code/CNN1_ex4.ipynb](https://drive.google.com/file/d/190UnS1etlKEyqRYwjxHnP4_jGYKXVT53/view) · cells 17–25

## ฝึกและเทียบผล

ฟังก์ชัน train_and_validate ใช้ loop เดียวกับทั้งสองโมเดล Cells 29–30 เรียกฝึก 20 Epochs กราฟ cell 33 เทียบ Learning Curves และ cells 35–37 ประเมินด้วย CrossEntropyLoss และ accuracy บน test Cells 40–41 เก็บภาพ Inference และป้ายทำนาย

> **อ้างอิง** [class04/code/CNN1_ex4.ipynb](https://drive.google.com/file/d/190UnS1etlKEyqRYwjxHnP4_jGYKXVT53/view) · cells 26–41

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 35

```python
outputs = model(images)
loss = criterion(outputs, labels)
preds = outputs.argmax(dim=1)
```

Loss รับ logits ส่วน argmax เปลี่ยน logits หนึ่งแถวเป็น class index หนึ่งค่า

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 36–37 บันทึก MLP 97.91% และ CNN 99.14% บน test ของการทดลองนี้ ค่า summary แสดง MLP 109,386 และ CNN 105,866 พารามิเตอร์ จึงเทียบผลในบริบทของรูปแบบการเชื่อมและวงจรฝึกชุดนี้


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** คำนวณ in_features ใหม่ถ้าตัด Pool ชั้นสองออก และอธิบายว่า Flatten ของ MLP ทำให้ตำแหน่งสัมพันธ์ถูกเก็บต่างจาก Feature Map ของ CNN อย่างไร

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md)
