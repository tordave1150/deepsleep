# Lab CNN2_ex3

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md) · [Index](../INDEX.md)

## เป้าหมาย

ดัดแปลง VGG16 จาก ImageNet เป็น CIFAR-10 และเทียบการเปลี่ยน Head กับการเปิดชั้นท้ายของ Backbone

ต้นฉบับ [class05/code/CNN2_ex3.ipynb](https://drive.google.com/file/d/1yZDfBqWsv2faZjDLtQQan6F5TnaKkqY0/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision transforms v2, torchinfo, torchview, NumPy, Matplotlib และ copy โค้ดดาวน์โหลด CIFAR-10 และ VGG16 weights กราฟ topology ใช้ renderer ของ torchview

## ปรับภาพและตาม Adaptive Pooling

CIFAR-10 มีภาพ RGB 32×32 ฝั่ง train ใช้ random crop และ horizontal flip ที่ขนาดเดิมก่อน Resize เป็น 224×224 และใช้ ImageNet mean/std ส่วน test ทำ Resize และ Normalize โดยไม่มี Augmentation ใน VGG16 ของ torchvision AdaptiveAvgPool ทำให้ feature size ก่อน Linear เป็น 512×7×7 ได้ในตัวอย่างทั้ง 224 และ 300 ส่วน LegacyVGG ที่ไม่มีขั้นนี้ส่งจำนวน Features ผิดเมื่อใช้ 300

> **อ้างอิง** [class05/code/CNN2_ex3.ipynb](https://drive.google.com/file/d/1yZDfBqWsv2faZjDLtQQan6F5TnaKkqY0/view) · cells 8–30

## เปรียบเทียบสามหัวและชุดพารามิเตอร์

vgg1 แทนเฉพาะ Linear ท้ายด้วย 4096→10 และเปิดฝึกเฉพาะชั้นนั้น vgg2 แทนหัวทั้งหมดด้วย Linear 25088→10 และตรึง Backbone vgg3 ใช้สำเนา vgg2 ที่ Weight แยกพื้นที่เก็บ แล้วเปิด features[28] ร่วมกับหัว สามแบบได้ logits (B,10) เหมือนกันและส่ง Gradient ให้คนละส่วน

> **อ้างอิง** [class05/code/CNN2_ex3.ipynb](https://drive.google.com/file/d/1yZDfBqWsv2faZjDLtQQan6F5TnaKkqY0/view) · cells 31–44

## ฝึก ตีความกราฟ และ Inference

รวบรวมเฉพาะพารามิเตอร์ที่ requires_grad เข้า Adam และฝึกแบบละ 15 Epochs ด้วย CrossEntropyLoss Cells 48–50 ส่ง test_loader เข้าอาร์กิวเมนต์ validation ตามที่ cell 55 ระบุเพื่อการสาธิต จึงใช้กราฟเปรียบเทียบภายใน Lab ได้ แต่ค่าชุดเดียวกันไม่ใช่ผลจากชุด test ที่แยกจากการสังเกตระหว่างฝึก Cell 61 แยก confidence ออกจาก correctness

> **อ้างอิง** [class05/code/CNN2_ex3.ipynb](https://drive.google.com/file/d/1yZDfBqWsv2faZjDLtQQan6F5TnaKkqY0/view) · cells 45–62

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 37

```python
vgg2.classifier = torch.nn.Linear(25088, 10)
```

หัวใหม่รับ 512×7×7 features หลัง Flatten แล้วให้สิบ logits ของ CIFAR-10

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 29 เก็บการลองขนาดภาพกับ LegacyVGG Cell 53 เก็บกราฟของสามแบบ และ cells 58–60 เก็บรูปตัวอย่างทำนาย ตัวเลข trainable parameters ใน summaries ต้องอ่านแยกจาก total parameters


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เขียนรายการ parameter groups ของ vgg3 แล้วทำนายว่ากลุ่มใดมี Gradient ถ้าจะใช้การเปรียบเทียบเพื่อเลือกโมเดล ให้กัน validation จากชุดฝึกและเก็บ test ไว้วัดปลายทาง

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md)
