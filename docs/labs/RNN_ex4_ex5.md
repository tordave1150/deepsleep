# Lab RNN_ex4_ex5

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md) · [Index](../INDEX.md)

## เป้าหมาย

ทำนายสามค่าพร้อมกันจากลำดับเดียว และแยก dense training loss จาก last-step metric

ต้นฉบับ [class06/code/RNN_ex4_ex5.ipynb](https://drive.google.com/file/d/1Qq7VZ2lE-_sSElLySGJvHHFZxjn0vZZ-/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ Python, NumPy, Matplotlib, PyTorch และ torchinfo พร้อมข้อมูล sine จำลองในหน่วยความจำ

## Target สองรูปแบบ

สร้าง sine มี Noise และแบ่งตามเวลา Window ยาว 50 และมี input หนึ่ง feature ต่อ step โหมด last คืนสามค่าหลัง Window เป็น (3,) โหมด dense สร้างสามค่าถัดไปสำหรับทุก timestep ใน Window เป็น (50,3) โดยเลื่อนตำแหน่ง target ตาม i ใน cell 12

> **อ้างอิง** [class06/code/RNN_ex4_ex5.ipynb](https://drive.google.com/file/d/1Qq7VZ2lE-_sSElLySGJvHHFZxjn0vZZ-/view) · cells 10–17

## Projection และ Loss

ทั้งสองโมเดลได้ RNN output (B,50,64) แบบ many-to-one ส่งเฉพาะ timestep สุดท้ายผ่าน Linear ให้ (B,3) ส่วน many-to-many ใช้ Linear เดียวกันทุก timestep ให้ (B,50,3) แบบ dense ใช้ MSE ทุก timestep ใน backward แต่แยกเก็บ MSE ของ [:,-1,:] เพื่อเทียบกับแบบ last ที่เห็นบริบทครบ

> **อ้างอิง** [class06/code/RNN_ex4_ex5.ipynb](https://drive.google.com/file/d/1Qq7VZ2lE-_sSElLySGJvHHFZxjn0vZZ-/view) · cells 18–26

## ประเมินบนสัญญาณจริง

กราฟ test เลือก Window จาก ground truth และวาด prediction สามจุดเป็นช่วง ๆ สำหรับ dense เลือกผล timestep สุดท้ายก่อนวาด ตัวอย่างไม่ป้อน forecast กลับไปต่อแบบ recursive

> **อ้างอิง** [class06/code/RNN_ex4_ex5.ipynb](https://drive.google.com/file/d/1Qq7VZ2lE-_sSElLySGJvHHFZxjn0vZZ-/view) · cells 27–32

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 19

```python
return self.fc(out[:, -1, :])
```

เลือก state หนึ่งตัวต่อลำดับ ส่วนโหมด dense ใน cell 20 ใช้ Linear บนทุก timestep

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 29 เก็บกราฟ learning curves ที่มี dense และ last-step metrics ส่วน cell 31 แสดงช่วงทำนายสามจุด ควรใช้ last-step metric คู่กันเมื่อต้องเทียบสองโมเดลบนโจทย์เดียว


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ถ้า Window มีสี่ timestep และทำนายสองค่า ให้เขียนดัชนีของ target สองแบบและระบุ slice ที่ทำให้ dense กับ last เทียบกันได้

[Lab Index](README.md) · [Lecture 6](../lectures/06-rnn.md)
