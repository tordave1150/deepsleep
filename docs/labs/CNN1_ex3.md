# Lab CNN1_ex3

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md) · [Index](../INDEX.md)

## เป้าหมาย

ตาม CNN ที่มีสอง inputs และสอง heads และดูว่า output แต่ละส่วนรับข้อมูลจากทางใด

ต้นฉบับ [class04/code/CNN1_ex3.ipynb](https://drive.google.com/file/d/1cT8fJo6miaqt8gZpCCFyWFpPcJf6Pv0K/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, PyTorch, torchinfo และ torchview พร้อม renderer สำหรับกราฟ ใช้ random tensors เพื่อสาธิต Forward ไม่มี Dataset หรือ Loss

## สองทางเข้าสู่ Feature Extractor

ภาพหนึ่ง channel และภาพสาม channels มีขนาด 224×224 แต่ใช้ extractor แยกกัน แต่ละสายผ่าน Conv ขนาด 3 ไม่มี padding และ MaxPool ขนาด 2 สองครั้ง จึงได้ 224→222→111→109→54 ในแต่ละแกน หลัง Flatten ได้ 32×54×54 = 93,312 Features ต่อสาย

> **อ้างอิง** [class04/code/CNN1_ex3.ipynb](https://drive.google.com/file/d/1cT8fJo6miaqt8gZpCCFyWFpPcJf6Pv0K/view) · cells 10–12

## แตกแขนงและรวม Features

Regression Head ใช้เฉพาะ Features ของภาพหนึ่ง channel และออก 4 ค่า ส่วน Classifier ต่อ features ของทั้งสองสายด้วย cat(dim=-1) จึงมี 186,624 ค่าเข้า Linear และได้ 20 logits การต่อที่แกน Batch จะเปลี่ยนจำนวนแถวและไม่ตรง Linear ที่กำหนด

> **อ้างอิง** [class04/code/CNN1_ex3.ipynb](https://drive.google.com/file/d/1cT8fJo6miaqt8gZpCCFyWFpPcJf6Pv0K/view) · cells 11–15

## ทดลอง Forward สองผลลัพธ์

สุ่ม inputs batch 7 ให้ตรง shape และ device ของโมเดล จาก forward ได้ tuple ผลคาดเป็น (7,4) และ (7,20) ตรวจ order ของสอง outputs ตอน unpack

> **อ้างอิง** [class04/code/CNN1_ex3.ipynb](https://drive.google.com/file/d/1cT8fJo6miaqt8gZpCCFyWFpPcJf6Pv0K/view) · cells 16–18

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 11

```python
x_concat = torch.cat((x_1ch, x_3ch), dim=-1)
y_regress = self.output_regress(x_1ch)
y_cls = self.output_cls(x_concat)
```

Classifier รับทั้งสอง paths ส่วน Regression ใช้เส้นทางหนึ่ง channel

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 14 แสดง summary 239,972,376 พารามิเตอร์ Cell 15 เก็บกราฟ SVG และ cell 18 พิมพ์ผลสอง heads ต้นทุนสูงส่วนหนึ่งมาจาก Linear ที่รับ flattened vector ขนาดใหญ่


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ลองไล่ tensor เมื่อสลับแกน concatenate จาก −1 เป็น 0 แล้วบอกว่า Linear จะเห็นจำนวน features เท่าใด

[Lab Index](README.md) · [Lecture 4](../lectures/04-cnn-basics.md)
