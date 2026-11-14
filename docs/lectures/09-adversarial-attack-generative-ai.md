# 9. Adversarial Attack และ Generative AI

[← ก่อนหน้า](08-attention-transformer.md) | [Index](../INDEX.md) | [ถัดไป →](10-autoencoder.md)

## แหล่งเรียน

class09 · ตรวจต้นฉบับ 2026-11-14

- [class09/09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view)
- [class09/09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view)

ผลสำรวจโฟลเดอร์ไม่พบ Notebook สำหรับ Class นี้

## เป้าหมายและพื้นฐาน

แยกวิธีเรียนเมื่อป้ายมีจำกัด อธิบายความเปราะบางของโมเดลและหลักการประเมิน Robustness จาก PDF แรก แล้วอ่านประวัติ Latent Space และการประเมิน Generative AI จาก PDF ที่สอง ทวน [Transfer Learning](05-cnn-classifier.md) และ [BERT กับ MAE](08-attention-transformer.md) ก่อนเริ่ม

## Knowledge Distillation และการเรียนด้วยป้ายน้อย

แผนภาพ Teacher–Student ให้ Student เรียนจาก soft predictions ของ Teacher ควบคู่กับสัญญาณจากป้ายจริง เพื่อถ่ายทอดข้อมูลว่า Teacher มองคลาสอื่นใกล้เคียงเพียงใด Weak Supervision ใช้ป้ายหยาบหรือไม่ครบ เช่น scribble บนภาพ โดย objective ให้ผลสอดคล้องกับหลักฐานที่มี Few-shot เริ่มจากตัวอย่างที่มีน้อยและ Zero-shot ในภาพ CLIP ใช้ความสัมพันธ์ระหว่างภาพกับคำอธิบายคลาสโดยไม่มีตัวอย่างติดป้ายของงานปลายทางในตัวอย่างนั้น

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ถ้า Anchor อยู่ห่าง Positive 0.3 และ Negative 0.9 โดย margin 0.4 ผลต่าง 0.3−0.9+0.4 เป็นลบ จึงได้ Triplet Loss เป็นศูนย์ตามกฎ max กับ 0 ถ้า Negative ขยับมา 0.5 จะได้ Loss 0.2 จุดประสงค์คือให้ Positive ใกล้กว่า Negative อย่างน้อยเท่า margin

> **อ้างอิง** [class09/09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 2–6

## Semi-supervised และ Self-supervised Learning

Semi-supervised Learning ใช้ป้ายที่มีอยู่ร่วมกับข้อมูลไม่ติดป้าย ภาพตัวอย่างให้โมเดลสร้าง pseudo-labels แล้วเรียนจากสองแหล่งร่วมกัน Self-supervised Learning สร้างสัญญาณการฝึกจากข้อมูลเอง เช่น ปิดบางส่วนแล้วทำนายสิ่งที่หาย หรือสร้างสองมุมมองของภาพเดียวกันเพื่อเรียน representation ภาพ pretraining → supervised fine-tuning แยกหน้าที่ของ pretext task กับงานเป้าหมาย

Contrastive Learning จัดมุมมองที่เกี่ยวข้องให้ representation ใกล้กัน ส่วน Masked Image Modeling ใช้ส่วนที่เห็นช่วยสร้างส่วนที่ปิด ผลเปรียบเทียบในสไลด์ระบุความต่างด้านการรักษารายละเอียดและการใช้งาน downstream ตามการทดลองที่ยกมา

> **อ้างอิง** [class09/09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 7–12

<a name="robustness"></a>

## ความเปราะบางและการป้องกัน

ภาพปัญหาใน ML Pipeline แยกความเสี่ยงตามจุดที่เกิด เช่น ข้อมูลฝึก การเปิดให้เรียกโมเดล และข้อมูลที่อาจอนุมานกลับได้ ภาพ Classifier แสดง input ที่มนุษย์รับรู้เหมือนเดิมแต่โมเดลเปลี่ยนผลทำนาย จึงควรประเมินความถูกต้องบนข้อมูลปกติและความทนทานต่อการเปลี่ยนแปลงแยกกัน ตัวอย่างในภาพจริงและข้อความ LLM ชี้ว่าความเปราะบางขึ้นกับ interface และสภาพการใช้งาน

ส่วนป้องกันอธิบาย Adversarial Training เป็นการฝึกเพื่อรับมืออินพุตที่จงใจรบกวน และ Defensive Distillation เป็นแนวคิดถ่ายทอดการกระจายผลจากโมเดลหนึ่งสู่อีกโมเดลหนึ่ง สไลด์ย้ำข้อจำกัดของความทนทานและผลกระทบต่อความแม่นยำ กรณี De-identification และการคุ้มครองงานศิลป์ยกเป้าหมายด้านความเป็นส่วนตัวและสิทธิ์ของผู้สร้าง

> **อ้างอิง** [class09/09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 12–22

## จาก Sequence Generation ถึง Neural Style Transfer

PDF ที่สองเริ่มจาก RNN ทำนาย token ถัดไปและส่งผลกลับเพื่อสร้างลำดับ แล้วเชื่อม Augmentation กับการสร้างรูปแบบที่แปรจากข้อมูลเดิม Neural Style Transfer ใช้ CNN ที่ฝึกแล้วสกัด Features ของ content image, style image และ generated image Content Loss เปรียบเทียบ Features ที่เก็บโครงภาพ ส่วน Style Loss เปรียบเทียบสถิติความสัมพันธ์ของ Feature Maps การวัด style จึงเน้นลักษณะร่วม เช่น texture มากกว่าตำแหน่ง Pixel ต่อ Pixel ภาพ DeepDream แสดงอีกแนวทางที่ขยาย pattern ที่กระตุ้น Feature ของโมเดล

> **อ้างอิง** [class09/09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 1–6

## จากงานวิจัยไปสู่การใช้งานกว้าง

ภาพตัวอย่างช่วงกลางแยก Image Synthesis, Image-to-Image Translation, Super-resolution และ Dataset Generation โดยถามว่าแต่ละงานเริ่มจาก noise, ภาพเงื่อนไข หรือข้อมูลชนิดอื่น ตัวอย่าง GReaT นำ Autoregressive Language Model มาใช้กับข้อมูลตารางที่มีคอลัมน์ต่างชนิดและมีปัญหาข้อมูลขาดหาย ส่วน Deepfakes ชี้ความเสี่ยงของภาพและเสียงที่ดูน่าเชื่อถือ

Timeline เชื่อม GAN/VAE กับ Diffusion, DALL-E, CLIP, Chatbot และ multimodal systems ภาพ Hype Cycle ใช้เล่าเรื่องการรับรู้และการนำไปใช้ในช่วงที่รายงาน การอยู่ตำแหน่งใดในกราฟไม่ได้ให้คะแนนคุณภาพโมเดล

> **อ้างอิง** [class09/09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 7–23

## Latent Space, Evaluation และข้อกังวล

แผนภาพ Latent Space แสดงตัวแทนที่เรียนรู้จากโครงสร้างข้อมูล ซึ่งนำไปใช้เพื่อจัดกลุ่มหรือให้ Decoder สร้างตัวอย่างได้ตามโมเดลที่เลือก การประเมินขึ้นกับชนิดผลลัพธ์ เช่น IS/FID/KID สำหรับภาพ และ BLEU/ROUGE สำหรับข้อความ ร่วมกับการประเมินของคน หากให้คนหลายคนอ่านผลให้พิจารณาความสอดคล้องของผู้ประเมินด้วย

สไลด์แยกมุมผู้ใช้ องค์กร และผู้กำกับดูแล โดยยก hallucination, ความลับของข้อมูล และสิทธิ์ในเนื้อหา ภาพท้ายยก detector กับ watermark เป็นวิธีช่วยสืบที่มาของเนื้อหา ซึ่งต้องประเมินแยกจากคุณภาพของงานที่สร้าง

> **อ้างอิง** [class09/09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 24–28

## ทบทวนและเชื่อมบท

เมื่อข้อมูลติดป้ายไม่พอ ให้ดูว่าการฝึกสร้าง supervisory signal จากที่ใด และเมื่อประเมิน Generative AI ให้เลือกตัววัดตามงาน [Class 10](10-autoencoder.md) ลงรายละเอียดว่า Latent Space ของ AE และ VAE ถูกสร้างอย่างไร

## คำถามทบทวน

1. Masked Image Modeling สร้าง target อย่างไรจากภาพที่ไม่มีป้าย
2. อธิบายความต่างระหว่าง Accuracy ในชุดปกติกับการประเมิน Robustness

[คำถามและเฉลย](../review/practice-and-solutions.md#class09) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| Knowledge Distillation และการเรียนด้วยป้ายน้อย | [09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 2–6 |
| Semi-supervised และ Self-supervised Learning | [09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 7–12 |
| ความเปราะบางและการป้องกัน | [09-1_Misc_AdversarialAttack_2handout_color.pdf](https://drive.google.com/file/d/1JHvVxHLdFKkjpGrfx5KL5D-YfOQqnWAx/view) · PDF หน้า 12–22 |
| จาก Sequence Generation ถึง Neural Style Transfer | [09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 1–6 |
| จากงานวิจัยไปสู่การใช้งานกว้าง | [09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 7–23 |
| Latent Space, Evaluation และข้อกังวล | [09-2_IntroGenerativeAI_2handout_color.pdf](https://drive.google.com/file/d/1CuaUQu0vakz9EWBEndSFT_QJaR5-C93l/view) · PDF หน้า 24–28 |

[← ก่อนหน้า](08-attention-transformer.md) | [Index](../INDEX.md) | [ถัดไป →](10-autoencoder.md)
