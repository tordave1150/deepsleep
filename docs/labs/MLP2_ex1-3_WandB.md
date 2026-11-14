# Lab MLP2_ex1-3_WandB

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md) · [Index](../INDEX.md)

## เป้าหมาย

อ่าน Sweep ของ Regression เดิมและตามเส้นทาง config → training → checkpoint → test

ต้นฉบับ [class03/code/MLP2_ex1-3_WandB.ipynb](https://drive.google.com/file/d/1b1dqdgBoF9OxYdPCJPlPpA8fIJTxfumz/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, NumPy, pandas, scikit-learn, Keras, torchinfo, Matplotlib, wandb และ python-dotenv ขั้น login, log, save และ sweep ติดต่อบริการภายนอกและอาจส่ง metrics หรือไฟล์ checkpoint ขึ้นบริการ Cell 11 อ่านค่า API key จาก environment

## ข้อมูลและโมเดล

แบ่ง Boston Housing แล้ว normalize 13 features จากสถิติ train ก่อนทำ TensorDataset Input (B,13) ผ่าน Linear/BatchNorm/ReLU/Dropout สองบล็อกและ Linear สุดท้ายออก (B,1) ให้จับคู่กับ target (B,1)

> **อ้างอิง** [class03/code/MLP2_ex1-3_WandB.ipynb](https://drive.google.com/file/d/1b1dqdgBoF9OxYdPCJPlPpA8fIJTxfumz/view) · cells 16–27

## Sweep และการฝึกต่อ run

Sweep สุ่ม batch size จาก 16/32/64, LR ระหว่าง 0.0001 ถึง 0.01 และ Adam หรือ SGD แต่ละ run ฝึก 50 Epochs และสร้างโมเดลใหม่หลังตั้ง seed เป้าหมายคือ best_val_mse ที่ต่ำที่สุดระหว่างฝึก เมื่อดีขึ้นโค้ดบันทึก checkpoint แล้วส่งด้วย wandb.save Cell 40 ขอรันห้าครั้ง โค้ดยังบันทึก test metric ต่อ run จึงต้องระวังการนำค่าเหล่านั้นกลับมาใช้เลือก config

> **อ้างอิง** [class03/code/MLP2_ex1-3_WandB.ipynb](https://drive.google.com/file/d/1b1dqdgBoF9OxYdPCJPlPpA8fIJTxfumz/view) · cells 28–40

## โหลด run ที่เลือก

ฟังก์ชันอ่าน best run จาก Sweep และดาวน์โหลดไฟล์โมเดล Cells 44–47 ประเมินและใช้โมเดลที่โหลด ข้อสังเกตจากโค้ดคือ model ที่สร้างใน train_sweep เป็นตัวแปรภายในฟังก์ชัน ส่วน cell 43 ใช้ model จากภายนอก จึงควรตรวจที่มาของ weights ก่อนตีความค่า last ว่าเป็นผลจาก Sweep

> **อ้างอิง** [class03/code/MLP2_ex1-3_WandB.ipynb](https://drive.google.com/file/d/1b1dqdgBoF9OxYdPCJPlPpA8fIJTxfumz/view) · cells 37, 41–47

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 40

```python
wandb.agent(sweep_id, function=train_sweep, count=5)
```

ให้บริการจัด config และเรียก train_sweep ตามจำนวน run ที่กำหนด

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 40 เก็บ log และ HTML ของ Sweep Cells 44–45 เก็บการโหลดและตัวเลขทดสอบ ข้อมูล dashboard แบบโต้ตอบขึ้นกับสิทธิ์ในบริการ


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เขียนตาราง config, best epoch, best_val_mse และ checkpoint ต่อ run แล้วอธิบายว่าทำไม last epoch อาจแพ้ best epoch โดยไม่ต้องเปิดบริการ

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md)
