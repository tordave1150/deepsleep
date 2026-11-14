# Lab CNN2_ex2

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md) · [Index](../INDEX.md)

## เป้าหมาย

ใช้ ResNet50 ที่ฝึกแล้วทำนาย ImageNet และตามจากผลดัชนีกลับไปที่ชื่อคลาส

ต้นฉบับ [class05/code/CNN2_ex2.ipynb](https://drive.google.com/file/d/1xGFLWEgDT9kVKxMsM8VmXvyxfGA-FxqT/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision, Pillow, NumPy, Matplotlib และ torchinfo โค้ดอาจดาวน์โหลด weights และเรียก wget เพื่อดึงภาพตัวอย่าง

## Weight, Preprocess และ Class Metadata

โหลด ResNet50_Weights.DEFAULT แล้วใช้ preprocessing และรายการ categories ที่ผูกกับ weight preset เดียวกัน รายการนั้นทำให้ดัชนี output ตรงกับป้ายที่ใช้ตอนฝึก

> **อ้างอิง** [class05/code/CNN2_ex2.ipynb](https://drive.google.com/file/d/1xGFLWEgDT9kVKxMsM8VmXvyxfGA-FxqT/view) · cells 8–10, 14–17

## เตรียมภาพและจัด Batch axis

อ่านภาพด้วย Pillow แล้วใช้ transform ของ weights เพื่อให้ขนาดและ normalization ตรงกับโมเดล เมื่อเพิ่ม Batch axis ได้ input (1,3,224,224) Output เป็น logits (1,1000) และ softmax(dim=1) แปลงคะแนนข้ามแกนคลาส

> **อ้างอิง** [class05/code/CNN2_ex2.ipynb](https://drive.google.com/file/d/1xGFLWEgDT9kVKxMsM8VmXvyxfGA-FxqT/view) · cells 11–15, 18

## อ่าน Top-k

ใช้ topk จำนวน 20 ดึง probability กับดัชนีในลำดับเดียวกัน แล้ว map กลับรายการ categories ตัวอย่างมีสุนัขมากกว่าหนึ่งตัว แต่ภาพทั้งรูปได้รับ distribution หนึ่งแถวสำหรับ Classification

> **อ้างอิง** [class05/code/CNN2_ex2.ipynb](https://drive.google.com/file/d/1xGFLWEgDT9kVKxMsM8VmXvyxfGA-FxqT/view) · cells 16–19

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 19

```python
top_prob, top_idx = torch.topk(output_prob, K)
```

top_prob กับ top_idx สอดคล้องทีละตำแหน่ง ใช้ดัชนีไปเปิดรายการ classes

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 13 เก็บภาพตัวอย่าง Cell 17 พิมพ์รายการคลาส Cell 19 พิมพ์ Top-20 ค่าที่อ่านเป็นคะแนนของภาพและ weight preset ใน Notebook


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ลด k เป็น 3 แล้วระบุว่าส่วนใดของผลเปลี่ยน หากชื่อคลาสดูผิดลำดับ ให้ตรวจว่าใช้ categories จาก weights เดียวกับ model

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md)
