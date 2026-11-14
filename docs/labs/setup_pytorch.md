# Lab setup_pytorch

[Lab Index](README.md) · [Lecture 1](../lectures/01-introduction-pytorch.md) · [Index](../INDEX.md)

## เป้าหมาย

ตรวจ environment ก่อนฝึกโมเดลและอ่าน shape จาก Tensor สุ่ม เชื่อมกับ Hardware Acceleration และ Tensor ใน Class 01

ต้นฉบับ [class01/setup_pytorch.ipynb](https://drive.google.com/file/d/1vuOWI98wHM9tpE5KYhdeUraSNOzKRa3u/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python 3, sys, NumPy, Matplotlib และ PyTorch คำสั่ง shell เรียก nvidia-smi ส่วน torch.accelerator และ torch.cuda ตรวจจาก Python ไม่มีขั้นโหลด Dataset หรือฝึกโมเดล

## ตรวจ NVIDIA และไลบรารี

บน Colab ข้อความใน cell 1 ให้เลือก GPU ผ่าน Runtime type เมื่อจำเป็น Cell 2 ใช้เครื่องมือ NVIDIA ดูว่า environment มองเห็น GPU หรือไม่ Cell 3 พิมพ์ version ของ Python, NumPy และ Matplotlib เพื่อใช้ประกอบเวลาตรวจความต่างระหว่าง environment

> **อ้างอิง** [class01/setup_pytorch.ipynb](https://drive.google.com/file/d/1vuOWI98wHM9tpE5KYhdeUraSNOzKRa3u/view) · cells 1–3

## ตรวจการรองรับจาก PyTorch

โค้ดถามจำนวน Accelerator ถ้าศูนย์จะรายงาน CPU ถ้ามีจึงอ่านชนิด Accelerator และถามเพิ่มว่า CUDA ใช้ได้หรือไม่ ค่า CUDA จากเครื่องมือ NVIDIA กับ torch.version.cuda มาจากคนละส่วน ข้อสังเกตจากโค้ดคือ num_gpus ถูกกำหนดในเงื่อนไข CUDA แต่ loop พิมพ์ชื่อ GPU อยู่ข้างนอก เมื่อมี Accelerator ชนิดอื่นและไม่มี CUDA ตัวแปรนี้อาจยังไม่มีค่า

> **อ้างอิง** [class01/setup_pytorch.ipynb](https://drive.google.com/file/d/1vuOWI98wHM9tpE5KYhdeUraSNOzKRa3u/view) · cell 4

## สร้างและอ่าน Tensor

ตั้ง seed 42 แล้วเรียก torch.randn(5, 3, 4) แกนแรกคือ Batch มีห้าตัวอย่าง แกนที่สองและสามเป็นแถวและคอลัมน์ การเลือก tensors[0] จึงเหลือ Matrix (3, 4) ส่วน tensors[0, 0, 0] เลือกค่าหนึ่งตำแหน่ง

> **อ้างอิง** [class01/setup_pytorch.ipynb](https://drive.google.com/file/d/1vuOWI98wHM9tpE5KYhdeUraSNOzKRa3u/view) · cell 5

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 5

```python
tensors = torch.randn(5, 3, 4)
```

ลำดับอาร์กิวเมนต์คือ Batch, แถว และคอลัมน์

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 3 รายงาน Python 3.12.13, NumPy 2.0.2 และ Matplotlib 3.10.0 Cell 4 รายงาน PyTorch 2.11.0+cu128 และ Tesla T4 Cell 5 แสดง torch.Size([5, 3, 4])


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เปลี่ยนแกนแรกเป็น 2 แล้วทำนาย shape ของ Tensor เต็มและตัวอย่างแรก ถ้า nvidia-smi ใช้ไม่ได้ ให้ดูผล import และการรายงาน CPU/Accelerator ใน cell 4 เพื่อระบุชั้นที่เกิดความต่าง

[Lab Index](README.md) · [Lecture 1](../lectures/01-introduction-pytorch.md)
