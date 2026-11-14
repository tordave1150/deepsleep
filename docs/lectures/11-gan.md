# 11. Generative Adversarial Network

[← ก่อนหน้า](10-autoencoder.md) | [Index](../INDEX.md)

## แหล่งเรียน

class11 · ตรวจต้นฉบับ 2026-11-14

- [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view)

ผลสำรวจโฟลเดอร์ไม่พบ Notebook สำหรับ Class นี้

## เป้าหมายและพื้นฐาน

เข้าใจ objective ของ Generator กับ Discriminator อธิบาย Mode Collapse และเลือกวิธีอ่านคุณภาพภาพที่สร้าง ทวน latent vector และ reconstruction จาก [Class 10](10-autoencoder.md) กับ classification loss จาก [Class 02](02-mlp-basics.md)

## Generator, Discriminator และงานที่ใช้

แผนภาพ GAN ให้ Generator G รับ noise z สร้างตัวอย่าง แล้วให้ Discriminator D ดูทั้งตัวอย่างจริงและตัวอย่างจาก G ฝั่ง D เรียนแยกสองแหล่ง ฝั่ง G เรียนจากคะแนนของ D ดังนั้นการเปลี่ยน Weight ของฝั่งหนึ่งเปลี่ยนโจทย์ของอีกฝั่งด้วย สไลด์เทียบกับ VAE ซึ่งมี Encoder และเงื่อนไขจัด Latent Space ชัดเจนกว่า

งาน Image Synthesis ใช้สร้างตัวอย่างใหม่ เช่น ภาพใบหน้า วัตถุ 3D หรือ Dataset งาน Translation มี input เงื่อนไข เช่น ภาพต้นฉบับหรือคำบรรยาย สไลด์ยังยก inpainting, super-resolution และงานที่ใช้ adversarial objective ช่วย Segmentation ตัวอย่าง Deepfakes แสดงผลกระทบต่อการเชื่อถือสื่อและการนำภาพบุคคลไปใช้

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 2–27

<a name="gan-objective"></a>

## Minimax และการสลับอัปเดต

ให้ $x$ เป็นตัวอย่างจริง, $z$ เป็น noise, $G(z)$ เป็นตัวอย่างที่สร้าง และ $D(\cdot)$ เป็นคะแนนความเป็นจริง Objective ใน handout คือ

$$
\min_G\max_D V(D,G)=\mathbb E_x[\log D(x)] + \mathbb E_z[\log(1-D(G(z)))].
$$

Expectation คือค่าเฉลี่ยภายใต้ distribution สไลด์อธิบายจากลูกเต๋า ซึ่งแต่ละค่าถ่วงด้วยโอกาสเกิด สำหรับการฝึกใช้ Batch ประมาณค่าเฉลี่ยสองเทอม D เพิ่มคะแนนตัวอย่างจริงและลดคะแนนตัวอย่างสร้าง ช่วงอัปเดต G ตรึงพารามิเตอร์ D แต่ใช้ Gradient ผ่าน D กลับไปหา G เพราะ D เป็นผู้ให้คะแนนตัวอย่างที่ G ผลิต

ช่วงต้นถ้า D ตัดสิน fake ได้มั่นใจมาก minimax term ฝั่ง G อาจส่งสัญญาณอ่อน สไลด์จึงยก non-saturating objective ที่ให้ G ลด $-\mathbb E_z\log D(G(z))$ คำนวณได้ด้วยเป้าป้ายจริงเมื่อปรับ G

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ถ้า D ให้ 0.1 แก่ภาพหนึ่ง ค่า $-\log 0.1$ สูงกว่า $-\log 0.7$ จึงให้ G มีแรงจูงใจสร้างผลที่ D ให้คะแนนสูงขึ้น

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 28–30

## Dynamic Optimization และ Mode Collapse

ในเกมสองฝั่ง การเปลี่ยน D ทำให้พื้นผิว Loss ของ G เปลี่ยนพร้อมกัน สไลด์ใช้ Nash equilibrium อธิบายจุดที่การเปลี่ยนฝั่งเดียวไม่ช่วยตน ในอุดมคติเมื่อ distribution ตรงกัน D ให้ 0.5 ได้ แต่ค่ากลางเดียวกันยังเกิดได้เมื่อ D เรียนไม่ขึ้น จึงต้องดูตัวอย่างและการกระจายประกอบ

ภาพ Mode Collapse แสดงหลาย noise inputs ให้ภาพคล้ายกันมาก แม้บางภาพดูสมจริง diversity ของทั้งชุดลดลง ตัวอย่าง Non-convergence ยังมี D ที่เก่งเกินจน G ขาด Gradient, การแกว่ง และการฝึกล้มเหลวทั้งสองฝั่ง สไลด์เทคนิคฝึกกล่าวถึง activation, noise distribution และ strided convolution เป็น heuristics ที่ต้องดูตาม setup

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 31–43

## IS, FID, KID และการอ่านตัวอย่าง

Inception Score ใช้ classifier อ่านความมั่นใจรายภาพและความหลากหลายของป้ายโดยรวม FID เปรียบเทียบค่าเฉลี่ยและ covariance ของ features จากภาพจริงกับภาพสร้าง ค่าต่ำแสดงว่ากลุ่ม features ใกล้กันตามตัวสกัดนั้น KID เป็นอีกตัววัดที่ต้นฉบับยกเปรียบเทียบ ตัวเลขเหล่านี้ไวต่อจำนวนภาพ ตัวสกัด Feature และ preprocessing จึงควรรายงานเงื่อนไขพร้อมกันและดูความหลากหลายกับภาพตัวอย่างโดยตรง Lab AE ใช้ Fréchet distance แบบ LeNet ซึ่งเทียบค่ากับมาตรฐาน Inception โดยตรงไม่ได้

ใน IS ให้ $p(y\mid x)$ เป็น distribution จาก classifier สำหรับภาพสร้างหนึ่งภาพ และ $p(y)$ เป็นค่าเฉลี่ยของ distributions เหล่านั้นทั้งชุด

$$
\mathrm{IS}=\exp\left(\mathbb E_x[D_\mathrm{KL}(p(y\mid x)\Vert p(y))]\right).
$$

หากภาพแต่ละภาพถูกจัดเข้าคลาสใดคลาสหนึ่งอย่างชัดเจน แต่ภาพต่าง ๆ กระจายข้ามหลายคลาส $p(y\mid x)$ จะกระจุกกว่าค่าเฉลี่ย $p(y)$ จึงช่วยให้คะแนนสูงขึ้น ถ้าทุกภาพกระจุกในคลาสเดียวกัน การกระจายรายภาพและค่าเฉลี่ยจะใกล้กันและส่วน KL ลดลง FID เพิ่มมุมเปรียบเทียบกับข้อมูลจริงผ่านสถิติ Features จึงควรอ่านคู่กับภาพและเงื่อนไขของตัวสกัด

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 37–40

## Conditional และ Image-to-Image GAN

DCGAN เปลี่ยน Generator/Discriminator เป็นโครง convolutional สำหรับภาพ cGAN ส่งเงื่อนไข เช่น class label ให้ทั้งสองฝั่ง Pix2Pix ใช้ภาพคู่จากสอง domain และใช้ทั้ง adversarial loss กับ L1 เพื่อรักษาความสอดคล้องกับภาพเป้าหมาย CycleGAN ใช้ภาพที่ไม่จับคู่แต่เพิ่ม cycle consistency ให้แปลง A → B → A แล้วกลับใกล้ input เดิม DiscoGAN ใช้การแปลงย้อนกลับเพื่อบังคับความสัมพันธ์ข้าม domain ในลักษณะใกล้เคียงกัน

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 44–50

## ProGAN, StyleGAN, StackGAN และ Diffusion

ProGAN เพิ่ม resolution ตามช่วงฝึกด้วยการค่อย ๆ เพิ่มบล็อก StyleGAN ใช้ mapping network และการควบคุมหลายสเกลเพื่อแยกปัจจัยที่มีผลต่อภาพ StyleGAN2 แก้ artifacts ที่สไลด์ชี้ให้เห็นในภาพและ StyleGAN3 จัดการการยึดกับพิกัด Pixel ส่วน StackGAN ใช้ข้อความเป็นเงื่อนไขและสร้างจากภาพหยาบไปละเอียด

ช่วงท้ายวางความเร็วในการสุ่ม คุณภาพ และความครอบคลุมของตัวอย่างไว้เป็นสามเกณฑ์ GAN มีจุดเด่นด้านการสร้างในครั้งเดียวแต่มีความท้าทายด้านเสถียรภาพ ความหลากหลาย และการควบคุม สไลด์เชื่อม DDPM, DDIM และ Latent Diffusion กับการเปลี่ยนวิธีสร้างภาพ และยก StyleGAN-T, GigaGAN และ DragGAN เป็นกรณีที่ยังใช้จุดเด่นของ GAN

> **อ้างอิง** [class11/11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 50–65

## ทบทวนและเชื่อมบท

ก่อนอ่านคะแนน GAN ให้ถามว่าภาพดูสมจริงหรือไม่ ครอบคลุมความหลากหลายเพียงใด และตอบเงื่อนไขที่กำหนดหรือไม่ แบบฝึกท้ายคอร์สให้แยกหน้าที่ของ G/D และอธิบายว่าทำไม loss ของเกมอย่างเดียวจึงไม่พอ

## คำถามทบทวน

1. เหตุใด G ยังต้องใช้ Gradient ผ่าน D ในช่วงที่ D ถูกตรึง
2. ถ้าภาพ GAN คมชัดแต่ซ้ำกันมาก จะใช้หลักฐานใดวินิจฉัย

[คำถามและเฉลย](../review/practice-and-solutions.md#class11) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| Generator, Discriminator และงานที่ใช้ | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 2–27 |
| Minimax และการสลับอัปเดต | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 28–30 |
| Dynamic Optimization และ Mode Collapse | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 31–43 |
| IS, FID, KID และการอ่านตัวอย่าง | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 37–40 |
| Conditional และ Image-to-Image GAN | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 44–50 |
| ProGAN, StyleGAN, StackGAN และ Diffusion | [11_GAN_2handout_color.pdf](https://drive.google.com/file/d/1MBwpnf-0IzmFSVXEWEPW28_OSe9w_3DO/view) · PDF หน้า 50–65 |

[← ก่อนหน้า](10-autoencoder.md) | [Index](../INDEX.md)
