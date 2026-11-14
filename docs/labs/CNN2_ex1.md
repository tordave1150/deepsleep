# Lab CNN2_ex1

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md) · [Index](../INDEX.md)

## เป้าหมาย

อ่านผลของ on-the-fly Augmentation ใน CNN สำหรับ MNIST แล้วตามว่า train กับ validation ได้ภาพคนละรูปแบบอย่างไร

ต้นฉบับ [class05/code/CNN2_ex1.ipynb](https://drive.google.com/file/d/1oWkr0u5yI0ige7-NqufDLmNS_cGQwzbC/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision transforms v2, NumPy, Matplotlib และ torchinfo มีการดาวน์โหลด MNIST และใช้ DataLoader workers ในชุดฝึก

## Transform สองแบบและ Split เดียว

Train ใช้ ToImage → RandomRotation ±45° → RandomAffine เลื่อนเล็กน้อย → float32 สเกลเป็น [0,1] → Normalize ส่วน validation/test ใช้ลำดับเดียวกันโดยตัดการสุ่มออก สร้าง dataset train และ validation ที่อ้างภาพต้นทางชุดเดียวกันแต่ใช้ transforms คนละชุด แล้วสุ่ม indices แยกกันเป็น 50,000 กับ 10,000 และตรวจ intersection

> **อ้างอิง** [class05/code/CNN2_ex1.ipynb](https://drive.google.com/file/d/1oWkr0u5yI0ige7-NqufDLmNS_cGQwzbC/view) · cells 9–17

## อ่านภาพซ้ำและประกอบ CNN

ฟังก์ชันดูตัวอย่างเรียก dataset[index] หลายครั้ง ภาพฝึกอาจเปลี่ยนเมื่อใช้ index เดิม ส่วนภาพ validation คงที่ โครง CNN ใช้ Conv ขนาด 3, Pool สองครั้ง และ Flatten (B,32,7,7) ไป (B,1568) จากนั้นส่ง Linear จนได้ logits (B,10)

> **อ้างอิง** [class05/code/CNN2_ex1.ipynb](https://drive.google.com/file/d/1oWkr0u5yI0ige7-NqufDLmNS_cGQwzbC/view) · cells 18–25

## Training และ Test

ฝึก 20 Epochs ด้วย Adam และ CrossEntropyLoss ดู Learning Curves ใน cell 31 และใช้ evaluate_model กับ test_loader ใน cell 34 รูป Inference ใน cell 37 วางผลทำนายและป้ายจริงคู่กันเพื่อเห็นตัวอย่างที่ผิด

> **อ้างอิง** [class05/code/CNN2_ex1.ipynb](https://drive.google.com/file/d/1oWkr0u5yI0ige7-NqufDLmNS_cGQwzbC/view) · cells 26–37

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 11

```python
T.RandomRotation(degrees=45),
T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
```

สองบรรทัดอยู่ใน Compose ฝั่ง train การสุ่มครั้งแรกเปลี่ยนมุมและครั้งที่สองเลื่อนตำแหน่ง

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 20–21 เก็บตัวอย่างจากการอ่านภาพซ้ำ Cell 34 บันทึก Test Loss 0.0360 และ Accuracy 98.81% ผลจาก Notebook คนละเล่มยังไม่ใช่ Ablation ที่ควบคุมการสุ่มทุกปัจจัย


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ลองวาด pipeline ของสองชุดข้อมูลแล้ววงขั้นที่ทำให้ index เดิมได้ภาพต่างกัน ถ้า validation มีภาพเปลี่ยนทุกครั้ง ให้ตรวจ Subset ว่าชี้ไปยัง dataset ที่ใช้ transform ใด

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md)
