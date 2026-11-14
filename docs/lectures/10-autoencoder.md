# 10. Autoencoder

[← ก่อนหน้า](09-adversarial-attack-generative-ai.md) | [Index](../INDEX.md) | [ถัดไป →](11-gan.md)

## แหล่งเรียน

class10 · ตรวจต้นฉบับ 2026-11-14

- [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view)
- Lab [AE_ex1](../labs/AE_ex1.md)

## เป้าหมายและพื้นฐาน

อธิบาย Reconstruction, Bottleneck และเหตุที่ VAE กำหนด distribution ใน Latent Space ได้ อ่าน BCE และ CNN จาก [Class 02](02-mlp-basics.md) กับ [Class 04](04-cnn-basics.md) ก่อนตาม Decoder ใน [AE_ex1](../labs/AE_ex1.md)

## Encoder, Bottleneck และ Reconstruction

ภาพ AE เริ่มจาก input x ที่ Encoder เปลี่ยนเป็น latent vector z แล้ว Decoder คืนภาพ x̂ ให้เทียบกับ x จึงใช้ input เป็น target ได้เอง ภาพสำหรับข้อมูล 0–1 ยก BCE ให้ตีความ output เป็นค่าต่อ Pixel ส่วน MSE/RMSE มองผลเป็น Regression Convolutional AE ลดขนาด Feature Maps ด้วย convolution และขยายคืนด้วย upsampling หรือ transposed convolution ประเด็นคือปลาย Decoder ต้องมี shape เท่ากับ target ก่อนคำนวณ Reconstruction Loss

> **อ้างอิง** [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 1–6

## ความจุและการประยุกต์ AE

ถ้า Encoder–Decoder มีความจุสูงอาจจำตัวอย่างได้แม้ Bottleneck เล็ก Denoising AE เปลี่ยน input ให้มี Noise หรือขาดบางส่วน แต่คง clean target ไว้ เพื่อให้โมเดลเรียนตัวแทนที่ช่วยคืนข้อมูล ภาพฝึกแบบซ้อนยกการฝึกทีละชั้นก่อนนำ Encoder ไปใช้ต่อ

Anomaly Detection ฝึกบนข้อมูลปกติและดู Reconstruction Error เมื่อเจอข้อมูลใหม่ Error สูงชี้ว่าภาพไม่คุ้นกับตัวแทนที่เรียนมา และการตั้ง threshold ยังต้องสัมพันธ์กับผลที่วัดจริง ส่วน Segmentation ใช้ Encoder–Decoder คล้ายกันแต่ target เป็น Mask จึงมีโจทย์ต่างจาก Reconstruction

> **อ้างอิง** [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 7–11

## Latent Space ของ AE และ VAE

Vanilla AE เรียนให้จุดของตัวอย่างที่เห็นถอดรหัสได้ดี ภาพจุดที่กระจายห่างกันชี้ว่าตำแหน่งระหว่างกลุ่มอาจไม่ decode เป็นภาพที่มีความหมาย คุณสมบัติที่สไลด์ใช้ประเมิน Latent Space ได้แก่ Continuity คือจุดใกล้กันมีผลใกล้เคียง และ Completeness คือพื้นที่ที่สุ่มภายใต้ distribution มีโอกาสได้ผลที่ใช้ได้

VAE ให้ Encoder ส่งค่าพารามิเตอร์ของ distribution ต่อ input หนึ่งตัวอย่าง ในกรณี Gaussian แบบแยกมิติใช้ mean และ variance ต่อมิติแล้ว sample z ให้ Decoder ภาพเปรียบเทียบจุดกับกลุ่มความน่าจะเป็นอธิบายว่าทำไมโมเดลต้อง decode จุดในละแวกใกล้กันด้วย

> **อ้างอิง** [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 12–16

<a name="vae-loss"></a>

## KL และ Reparameterization

เมื่อ posterior ใช้ Gaussian แบบ diagonal และ prior เป็น $\mathcal N(0,I)$ ค่า KL ของหนึ่งตัวอย่างเขียนได้ว่า

$$
L_\mathrm{KL}=-\frac12\sum_{j=1}^{d}\left(1+\log \sigma_j^2-\sigma_j^2-\mu_j^2\right).
$$

$j$ ระบุมิติ latent, $\mu_j$ คือ mean และ $\sigma_j$ คือ standard deviation ถ้า $\mu_j=0,\sigma_j=1$ ทุกมิติจะได้ KL เป็น 0 ตัวอย่าง Lab ใช้ $L=L_\mathrm{recon}+\beta L_\mathrm{KL}$ เพื่อปรับแรงของ regularization เมื่อ $\beta$ สูงเกินบริบทของ Loss ตัวแทนอาจละเลยรายละเอียดของ input ถ้าต่ำเกินไป latent อาจไม่สอดคล้องกับ prior ที่ใช้สุ่ม

ให้ $\epsilon\sim\mathcal N(0,I)$ แล้วคำนวณ $z=\mu+\sigma\odot\epsilon$ การสุ่มอยู่ใน $\epsilon$ ขณะที่เส้นทางจาก $z$ ไปยัง $\mu,\sigma$ อยู่ใน Computation Graph ใน PyTorch หาก Encoder ให้ logvar คำนวณ $\sigma=\exp(\tfrac12\operatorname{logvar})$ ก่อนคูณ

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ในมิติเดียวถ้า $\mu=2,\sigma=0.5,\epsilon=-1$ ได้ $z=1.5$ ถ้าสุ่ม $\epsilon$ ใหม่ z เปลี่ยนได้แม้ input เดิม

> **อ้างอิง** [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 17–20

> **อ้างอิง** [class10/AE_ex1.ipynb](https://drive.google.com/file/d/147T5laBtRsXa6WI2BxLGBbb9bYDbER93/view) · cells 31–32, 38–39

## Interpolation และข้อจำกัดด้านภาพ

การ morph ใช้จุดระหว่าง $z_A$ และ $z_B$ ด้วยน้ำหนัก $1-\alpha$ กับ $\alpha$ ภาพต่อเนื่องช่วยตรวจว่าการเดินใน latent ส่งผลอย่างไร ตัวอย่าง Latent Arithmetic ใช้ทิศของ feature ปรับ attribute ส่วน AE_ex1 วาดทั้ง scatter plot ของ latent codes และกริดภาพที่ decode ตามตำแหน่ง จึงเห็นได้ว่าพื้นที่ที่ Encoder ใช้จริงตรงกับตำแหน่งที่ลองสุ่มเพียงใด

สไลด์อภิปรายภาพที่นุ่มหรือเบลอเมื่อ Decoder ถูกจูงให้เลือกค่าปลอดภัยสำหรับรายละเอียดที่แปรมาก และยก Photo Restoration กับการบ้าน DIY Image Generator เพื่อทดสอบ AE/VAE ในบริบทต่างกัน

> **อ้างอิง** [class10/10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 21–27

## ทบทวนและเชื่อมบท

วัด Reconstruction จาก input เดิม วัดโครงสร้าง latent จากการกระจายและ interpolation และวัดการสุ่มด้วยตัวอย่างที่สุ่มจริง [Class 11](11-gan.md) เปลี่ยน objective จากการเทียบ input กับ reconstruction ไปเป็นการให้ Discriminator ส่งสัญญาณกลับ

## คำถามทบทวน

1. คำนวณ KL เมื่อ mean และ logvar เป็นศูนย์
2. ทำไม latent ที่ reconstruct ได้ดีอาจใช้สุ่มภาพได้ไม่ทั่ว

[คำถามและเฉลย](../review/practice-and-solutions.md#class10) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| Encoder, Bottleneck และ Reconstruction | [10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 1–6 |
| ความจุและการประยุกต์ AE | [10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 7–11 |
| Latent Space ของ AE และ VAE | [10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 12–16 |
| KL และ Reparameterization | [10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 17–20 |
| Interpolation และข้อจำกัดด้านภาพ | [10_AutoEncoder_2handout_color.pdf](https://drive.google.com/file/d/1IhFZPA8z8ONEHHPo4dEj7Q4fTfxOz0FI/view) · PDF หน้า 21–27 |

[← ก่อนหน้า](09-adversarial-attack-generative-ai.md) | [Index](../INDEX.md) | [ถัดไป →](11-gan.md)
