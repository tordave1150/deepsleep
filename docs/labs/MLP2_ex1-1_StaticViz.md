# Lab MLP2_ex1-1_StaticViz

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md) · [Index](../INDEX.md)

## เป้าหมาย

ฝึก MLP Regression และใช้กราฟหลังฝึกกับ Checkpoint ที่เลือกจาก validation

ต้นฉบับ [class03/code/MLP2_ex1-1_StaticViz.ipynb](https://drive.google.com/file/d/14hvRoGYzQ9DRh2IFoNODLto5NuSGobRL/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, pandas, scikit-learn, Keras สำหรับ Boston Housing, PyTorch, torchinfo และ Matplotlib โค้ดเลือก CUDA, MPS หรือ CPU และอาจดาวน์โหลด Dataset กับเขียน Checkpoint ในเครื่อง

## แยกชุดและเตรียม Tensor

โหลด Boston Housing จาก Keras แล้วแบ่งชุดที่ Keras แยกไว้เป็น validation/test ครึ่งต่อครึ่งก่อนหา mean กับ standard deviation จาก train เพียงชุดเดียว ใช้สถิติเดียวกันกับทั้งสามชุดเพื่อหลีกเลี่ยงข้อมูลรั่ว Input เป็น (B,13) และ target ราคาถูก view(-1,1) ให้ตรงกับผล Regression

> **อ้างอิง** [class03/code/MLP2_ex1-1_StaticViz.ipynb](https://drive.google.com/file/d/14hvRoGYzQ9DRh2IFoNODLto5NuSGobRL/view) · cells 9–16

## โมเดลและ Checkpoint

BostonModel ใช้ Linear 13→64→32→1 ชั้น Hidden ผ่าน BatchNorm1d, ReLU และ Dropout 0.2 หรือ 0.1 Cell 23 ให้เลือกระหว่างเก็บ weights กับ checkpoint ที่มี optimizer และ epoch โค้ดโหลดสร้างโมเดลจากโครงเดิมเพื่อใส่ state_dict แล้วค่อยย้ายไป device

> **อ้างอิง** [class03/code/MLP2_ex1-1_StaticViz.ipynb](https://drive.google.com/file/d/14hvRoGYzQ9DRh2IFoNODLto5NuSGobRL/view) · cells 17–27

## ฝึก วาดกราฟ และทดสอบ

ใช้ Adam, batch size 32 และ MSE เป็น Loss ส่วน MAE ใช้รายงาน เก็บค่าเฉลี่ยถ่วงตามจำนวนตัวอย่างใน Batch แล้วบันทึกเมื่อ validation MSE ต่ำลง Cell 31 วาดคู่ train/val หลังฝึก Cells 33–35 เทียบ last กับ best ใน test และ cell 37 ทำ Inference ด้วย best model

> **อ้างอิง** [class03/code/MLP2_ex1-1_StaticViz.ipynb](https://drive.google.com/file/d/14hvRoGYzQ9DRh2IFoNODLto5NuSGobRL/view) · cells 28–37

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 29

```python
if val_mse < best_val_loss:
    best_val_loss = val_mse
    save_model(model, optimizer, epoch)
```

บันทึกเมื่อ MSE ของ validation ต่ำกว่าค่าต่ำสุดเดิม

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 20 แสดง 3,201 พารามิเตอร์ Cell 31 เก็บกราฟ MSE และ MAE Cell 35 รายงาน test MSE ของ best validation checkpoint ที่ 20.8666 และ last checkpoint ที่ 18.2829 ขณะที่ test MAE เป็น 2.7599 และ 2.8066 ตามลำดับ ทั้ง checkpoint และ Metric ให้ข้อสรุปต่างกันได้ ค่าต่ำสุดของ validation เป็นกฎเลือกที่กำหนดไว้ก่อนดู test


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ลองเทียบ Epoch ของ val MSE ต่ำสุดกับ Epoch สุดท้าย หากเกิด shape mismatch ให้ตรวจว่า target เป็น (B,1) เหมือน prediction และถ้าจะเทียบ train/val ให้พิจารณาว่า Dropout ทำงานเฉพาะ train

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md)
