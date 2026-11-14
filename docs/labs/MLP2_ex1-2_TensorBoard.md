# Lab MLP2_ex1-2_TensorBoard

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md) · [Index](../INDEX.md)

## เป้าหมาย

ติดตาม train/validation MSE และ MAE ตาม Epoch พร้อมเทียบกับกราฟที่วาดหลังฝึก

ต้นฉบับ [class03/code/MLP2_ex1-2_TensorBoard.ipynb](https://drive.google.com/file/d/1oSoE_eiPR6WgtIZTWfGwBWAHVDS7rIvr/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

เพิ่ม TensorBoard และ SummaryWriter จาก torch.utils.tensorboard จากชุด NumPy, pandas, scikit-learn, Keras, PyTorch, Matplotlib และ torchinfo ใช้ Notebook magics สำหรับเปิด dashboard และเขียนไฟล์ใต้ runs

## Dataset และ Regression

แบ่ง Boston Housing เป็น train/val/test ก่อน standardize 13 features ด้วยสถิติ train แล้วจัด target เป็นคอลัมน์เดียว โมเดล 13→64→32→1 วาง BatchNorm กับ Dropout ใน Hidden Layers จึงควรตาม shape (B,13) ไป (B,1) ก่อนอ่าน MSE

> **อ้างอิง** [class03/code/MLP2_ex1-2_TensorBoard.ipynb](https://drive.google.com/file/d/1oSoE_eiPR6WgtIZTWfGwBWAHVDS7rIvr/view) · cells 12–23

## Event logs และ Checkpoint

ตัวบันทึกเขียนใต้ runs/my_experiment ใน cell 35 ที่จบ Epoch ใช้ add_scalars จัด train กับ val คู่กันภายใต้ Loss/MSE และ Loss/MAE แล้ว flush เพื่อส่งข้อมูลไปยังไฟล์ Cell 33 เปิด TensorBoard โดยอ่านไดเรกทอรีแม่ runs ระหว่างนั้น training ใช้ Adam และบันทึก checkpoint เมื่อ validation MSE ลดลง

> **อ้างอิง** [class03/code/MLP2_ex1-2_TensorBoard.ipynb](https://drive.google.com/file/d/1oSoE_eiPR6WgtIZTWfGwBWAHVDS7rIvr/view) · cells 24–35

## อ่านผลย้อนหลัง

กราฟ Matplotlib อ่าน history ชุดเดียวกับที่ส่งไป SummaryWriter ตรวจว่ากราฟใช้ Epoch ที่ตรงกัน และว่าโหลด best ก่อนนำไป Inference Cells 39–41 ประเมินทั้ง last และ best บน test โดย MSE กับ MAE มาจาก residual ชุดเดียวกันแต่ให้น้ำหนัก error ใหญ่ต่างกัน

> **อ้างอิง** [class03/code/MLP2_ex1-2_TensorBoard.ipynb](https://drive.google.com/file/d/1oSoE_eiPR6WgtIZTWfGwBWAHVDS7rIvr/view) · cells 36–43

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 35

```python
writer.add_scalars('Loss/MSE', {
    'train': history['train_mse'][-1],
    'val': history['val_mse'][-1]
}, epoch)
```

ค่า train และ val ใช้ x-axis เดียวกันเพราะส่ง epoch เดียวกัน

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 33 มี output HTML ของ TensorBoard และ cell 37 เก็บภาพกราฟนิ่ง Cell 41 เก็บการเปรียบเทียบ last/best บน test การเปิด dashboard จาก HTML ที่บันทึกอาจต้องใช้ Runtime ที่สร้าง logs


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ระบุ log directory แยกต่อ run แล้วทำนายว่าการอ่านจาก runs จะเห็น tag ซ้ำจากกี่ run ถ้ากราฟว่างให้ตรวจเส้นทางเดียวกันกับ SummaryWriter และการ flush ก่อนตรวจโมเดล

[Lab Index](README.md) · [Lecture 3](../lectures/03-mlp-problems-visualization.md)
