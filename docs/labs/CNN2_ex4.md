# Lab CNN2_ex4

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md) · [Index](../INDEX.md)

## เป้าหมาย

สร้าง Grad-CAM สำหรับ VGG16 ด้วยไลบรารี Hooks และ FX แล้วดูว่าทั้งสามทางดึง Activation กับ Gradient จากไหน

ต้นฉบับ [class05/code/CNN2_ex4.ipynb](https://drive.google.com/file/d/1sYh6d32Q7E1296m15XNXq1UdDGjWVzyT/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision, torchinfo, NumPy, Matplotlib, OpenCV, Pillow, requests และ grad-cam ดึงภาพและ pretrained weights ผ่าน network ตามโค้ด

## สองรูปแบบของ Input

VGG16 อยู่ใน eval mode ภาพสำหรับโมเดลผ่าน preprocessing ของ Weight แล้วเพิ่ม Batch axis เป็น (1,3,224,224) ส่วน rgb_img เก็บ HWC ค่า 0 ถึง 1 สำหรับซ้อนสีของ heatmap ทั้งสองมาจากภาพเดียวกันและใช้คนละหน้าที่

> **อ้างอิง** [class05/code/CNN2_ex4.ipynb](https://drive.google.com/file/d/1sYh6d32Q7E1296m15XNXq1UdDGjWVzyT/view) · cells 9–18

## คำนวณ Heatmap สามวิธี

ทางไลบรารีส่ง target layer และ class index ให้ GradCAM ทาง Hooks บันทึก activation จาก forward และ gradient จาก backward แล้วถอด handles ทาง FX ขอทั้ง target_features และ output จาก graph ใหม่ ใช้ retain_grad กับ activation ก่อน backward ของ logit ที่เลือก จาก Gradient shape (1,C,H,W) เฉลี่ยแกน H,W เป็น (1,C,1,1) แล้วคูณและรวม channel เหลือแผนที่ H×W ก่อน ReLU และ resize

> **อ้างอิง** [class05/code/CNN2_ex4.ipynb](https://drive.google.com/file/d/1sYh6d32Q7E1296m15XNXq1UdDGjWVzyT/view) · cells 19–31

## เปรียบเทียบตามคลาส

คำนวณ Top-10 คลาสก่อนวนเรียกสามวิธีด้วย class index เดียวกันต่อแถว ดู target และตำแหน่งที่สว่างเทียบกัน หากปิด autograd ในเส้นทาง Grad-CAM จะไม่มี Gradient แม้โมเดลอยู่ใน eval mode

> **อ้างอิง** [class05/code/CNN2_ex4.ipynb](https://drive.google.com/file/d/1sYh6d32Q7E1296m15XNXq1UdDGjWVzyT/view) · cells 32–34

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 31

```python
weights = torch.mean(grads, dim=(2, 3), keepdim=True)
mask = torch.sum(weights * activations, dim=1).squeeze()
```

keepdim ทำให้ Weight broadcast บนทั้งแผนที่ของ channel นั้น จากนั้น sum รวมทุก channel

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 15 และ 17 เก็บภาพและ shape input Cell 33 มีรายการคลาส ส่วน cell 34 มี stream log และโค้ดสร้างการเปรียบเทียบภาพ ให้ยึดรูปแบบ output ที่บันทึกจริงและใช้ source เพื่ออ่านภาพที่ Runtime แสดง


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ติดตามตำแหน่งที่เก็บ gradients ของ Hooks และ FX แล้วอธิบายว่าทำไมค่า weights ต่อ channel มี shape (1,C,1,1) หากทำหลายครั้ง ให้ตรวจการถอด Hooks และการจับคู่ขนาด overlay

[Lab Index](README.md) · [Lecture 5](../lectures/05-cnn-classifier.md)
