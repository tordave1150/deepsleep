# Lab LSTM_ex1

[Lab Index](README.md) · [Lecture 7](../lectures/07-lstm-gru.md) · [Index](../INDEX.md)

## เป้าหมาย

เปรียบเทียบ RNN/LSTM/GRU บน simple และ composite sine แล้วอ่าน one-step แยกจาก recursive forecast

ต้นฉบับ [class07/code/LSTM_ex1.ipynb](https://drive.google.com/file/d/15JRUyGd2e2U1arCPnTniFhZek5oPafu2/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, Matplotlib, PyTorch และ torchinfo มี CUDA/MPS/CPU และฝึกหกโมเดลจากข้อมูลจำลอง

## สัญญาณและการคุมการสุ่ม

ใช้สัญญาณ 5,000 จุดและ Window 250 ค่าแบบ simple sine และผลรวม sine สองความถี่ผสม Noise แบ่งตามเวลา ในการเปรียบเทียบมี generator แยกสำหรับ shuffle ของ DataLoader เพื่อคุมลำดับ Batch ข้าม Architectures

> **อ้างอิง** [class07/code/LSTM_ex1.ipynb](https://drive.google.com/file/d/15JRUyGd2e2U1arCPnTniFhZek5oPafu2/view) · cells 10–16, 21–23

## สลับ Cell และเก็บ Snapshots

RecurrentModel สลับ nn.RNN, nn.LSTM หรือ nn.GRU โดยหัวอ่าน out[:,-1,:] จาก (B,250,128) เพื่อทำนาย (B,1) ใช้ MSE กับ Adam และ clipping ตาม training loop เก็บ recursive forecasts ที่ Epoch 10,25,50,75,100 เพื่อเห็นพฤติกรรมเปลี่ยนตามช่วงฝึก

> **อ้างอิง** [class07/code/LSTM_ex1.ipynb](https://drive.google.com/file/d/15JRUyGd2e2U1arCPnTniFhZek5oPafu2/view) · cells 15–22

## อ่านหกการทดลอง

Cell 30 ใช้ Window ค่าจริง Cell 32 ป้อนผลทำนายกลับเข้า Window และ cell 34 เปรียบเทียบ snapshots สำหรับ composite signal หาก MSE one-step ต่ำแต่ recursive หลุดคาบ ต้องพิจารณาความต่างของอินพุตและ objective ที่ใช้ฝึก

> **อ้างอิง** [class07/code/LSTM_ex1.ipynb](https://drive.google.com/file/d/15JRUyGd2e2U1arCPnTniFhZek5oPafu2/view) · cells 24–35

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 20

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

อยู่หลัง backward และก่อน step จึงจำกัด Gradient ที่จะนำไปใช้จริง

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 30, 32 และ 34 เก็บกราฟตามสามมุมมอง Cell 35 อภิปรายว่าใน run นี้ GRU ตามสัญญาณ recursive ได้ใกล้ที่สุดในบริบทที่แสดง ผลขึ้นกับ seed, Window และขนาดโมเดล


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เลือก Epoch จาก one-step loss แล้วเทียบกับ Epoch ที่เลือกจากความคลาดเคลื่อนช่วง recursive ให้สังเกตกรณีที่อันดับต่างกัน

[Lab Index](README.md) · [Lecture 7](../lectures/07-lstm-gru.md)
