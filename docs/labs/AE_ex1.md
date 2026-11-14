# Lab AE_ex1

[Lab Index](README.md) · [Lecture 10](../lectures/10-autoencoder.md) · [Index](../INDEX.md)

## เป้าหมาย

เทียบ Standard AE, KL-dominated VAE และ Balanced VAE บน FashionMNIST โดยตามทั้งภาพ Reconstruction และตำแหน่งที่สุ่มใน Latent Space

ต้นฉบับ [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, torchvision, NumPy, Matplotlib, torchinfo และ torchmetrics มีการโหลด FashionMNIST และฝึก LeNet judge แยกจาก AE/VAE โค้ดเลือก CUDA, MPS หรือ CPU

## Dataset และ LeNet Judge

FashionMNIST มีภาพเทา 28×28 แปลงเป็น float32 ในช่วง 0–1 แล้วแบ่ง train 50,000 และ validation 10,000 ด้วย random_split พร้อมแยก test ไว้ Cell 21 สร้าง LeNet5 ให้ทำนาย 10 คลาสเพื่อเรียน Features หลังฝึกประเมิน test และตรึงพารามิเตอร์ judge ใน AE/VAE ป้ายช่วยระบายสี scatter plot และใช้ฝึก judge ส่วน Reconstruction ใช้ภาพเป็น target

> **อ้างอิง** [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · cells 10–29

## AE, VAE และ Tensor Shapes

Conv2d stride 2 สองชั้นเปลี่ยน (B,1,28,28) → (B,32,14,14) → (B,64,7,7) หลัง flatten ได้ (B,3136) AE สร้าง z สองค่า ส่วน VAE สร้าง mu และ logvar อย่างละ (B,2) แล้ว sample z Decoder ขยายกลับเป็น (B,64,7,7) และใช้ transposed conv ให้ได้ (B,1,28,28) ที่ลงท้าย Sigmoid

BCE รวมทุก Pixel แล้วหารด้วย Batch Size ส่วน KL รวมสอง latent dimensions ต่อภาพก่อนเฉลี่ยทั้ง Batch Cell 32 คืน total, BCE และ KL แยกเพื่ออ่านผลของ beta

> **อ้างอิง** [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · cells 30–32

## การทดลองและ Fréchet Distance

ใช้ beta 0, 250 และ 5 สำหรับสาม experiments ฝึก 15 Epochs และเก็บภาพที่ Epoch 1,5,10,15 metric ใช้ LeNet features เปรียบภาพจริงกับ reconstruction แบบกลุ่ม ดังนั้นเป็น LeNet-based Fréchet distance และ VAE ยังสุ่ม z ใน forward ตอนประเมินด้วย Total Loss ระหว่าง models ใช้ beta ต่างกันจึงควรอ่าน Reconstruction และ metric แยก

> **อ้างอิง** [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · cells 33–44

## สุ่มและดู Latent Space

ใช้ Gaussian latent สองมิติชุดเดียวกัน decode ผ่านสาม models แล้วเปรียบเทียบ Cell 49 ใช้ deterministic z ของ AE หรือ posterior mean ของ VAE ทำ scatter ส่วน cell 54 สร้างกริด 20×20 เพื่อดูว่าพื้นที่ต่าง ๆ decode เป็นอะไร ภาพสุ่ม, ภาพ reconstruct และ scatter ให้หลักฐานต่างกัน

> **อ้างอิง** [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · cells 45–55

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 31

```python
return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
```

randn_like ให้ Noise shape เดียวกับ mu ส่วน exp ครึ่ง logvar ให้ standard deviation

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cell 29 เก็บผล judge Cells 41 และ 44 เก็บกราฟของแต่ละรุ่นและกราฟระยะเปรียบเทียบ ใน cell 44 เส้นของ KL-dominated VAE สูงกว่าสองแบบอื่นชัดเจน ใน cell 47 ภาพสุ่มจากรุ่นเดียวกันยังคล้ายกันมากแม้ใช้ latent vector ต่างกัน ส่วน scatter ใน cells 50–52 แสดง posterior means กระจุกชิดกันและ grid ใน cell 55 แสดงภาพที่เปลี่ยนน้อยในหลายตำแหน่ง หลักฐานเหล่านี้สอดคล้องกับการที่ KL มีน้ำหนักมากในการทดลองนี้


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ไล่ shape หลัง ConvTranspose2d แต่ละชั้น และอธิบายว่าทำไมตัวหารของ BCE เป็น Batch Size หาก scatter สวยแต่ภาพสุ่มไม่ดี ให้ตรวจว่าพื้นที่ที่สุ่มตรงกับตำแหน่งที่ Encoder ใช้จริงหรือไม่

[Lab Index](README.md) · [Lecture 10](../lectures/10-autoencoder.md)
