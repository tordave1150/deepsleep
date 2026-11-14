# 5. CNN สำหรับ Image Classification

[← ก่อนหน้า](04-cnn-basics.md) | [Index](../INDEX.md) | [ถัดไป →](06-rnn.md)

## แหล่งเรียน

class05 · ตรวจต้นฉบับ 2026-11-14

- [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view)
- Lab [CNN2_ex1](../labs/CNN2_ex1.md)
- Lab [CNN2_ex2](../labs/CNN2_ex2.md)
- Lab [CNN2_ex3](../labs/CNN2_ex3.md)
- Lab [CNN2_ex4](../labs/CNN2_ex4.md)

## เป้าหมายและพื้นฐาน

เปรียบเทียบทางเลือกใน CNN Image Classifier ตั้งแต่ Augmentation และ Backbone ไปถึง Transfer Learning และการอ่าน Grad-CAM ทวน Convolution, Pooling และ channels จาก [Class 04](04-cnn-basics.md)

## จาก LeNet และ AlexNet สู่ Data Augmentation

ภาพสถาปัตยกรรมเริ่มจาก ConvNet และ LeNet-5 สำหรับตัวเลข ไปถึง AlexNet ใน ImageNet โดยแยกส่วนสกัด Feature ออกจากตัวตัดสินคลาส AlexNet ยก Dropout และ Data Augmentation มาใช้กับการฝึก สไลด์ Augmentation แสดงภาพต้นฉบับหนึ่งรูปที่กลายเป็นหลายรูปผ่าน crop, สี, มุมมอง หรือ scale โดยเลือก Transformation ที่ยังเหมาะกับ label ของโจทย์

On-the-fly Transform ทำให้ตัวอย่างเดียวเปลี่ยนได้ระหว่าง Epoch และ Composition กำหนดลำดับการแปลง Scale Jittering เปลี่ยนขนาด ส่วน Copy-Paste ในตัวอย่าง Segmentation ต้องจัดการทั้งรูปและ Mask CutMix กับ MixUp รวมตัวอย่างและป้ายที่ระดับ Batch ส่วน Test-Time Augmentation รวมการทำนายของรูปหลายเวอร์ชัน ขั้นเหล่านี้มีบทบาทต่างกันตามเวลาที่ใช้

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 2–14

## VGG, Inception, ResNet และตระกูลอื่น

VGG ใช้การซ้อน 3×3 Conv เพื่อสร้างขอบเขตการมองที่กว้างขึ้นและแทรก Nonlinearity ระหว่างชั้น ภาพ FC ↔ Conv แสดงว่าหัวที่ Flatten ขนาดคงที่ติดกับจำนวน Features และสามารถเขียนใหม่ด้วย Convolution ที่มี Kernel ครอบพื้นที่เดิมเพื่อไล่คำนวณบนภาพใหญ่ขึ้น Inception ใช้แขนงหลายขนาดและ 1×1 Conv ลดจำนวน channels ก่อนการคำนวณที่แพง

ResNet วางทางลัดบวก input ให้ชั้นเรียน residual ภาพ DenseNet ต่อ Feature Maps จากชั้นก่อนหน้าด้วย concatenation ซึ่งต่างจากการบวก Xception แยกการกรองราย channel ออกจากการผสม channels ด้วย 1×1 NASNet ยกการค้นหา Architecture และ EfficientNet ปรับ depth, width กับ resolution ร่วมกัน สไลด์ยังยก RepVGG และ NFNets เพื่อแสดงทางเลือกการออกแบบ ควรอ่าน Accuracy คู่กับ FLOPs, RAM, Inference time และพลังงานตามฮาร์ดแวร์ที่ทดสอบ

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 15–28

## Transfer Learning และการเลือกชั้น

Pretraining บน upstream task ให้ Backbone เริ่มจาก Weight ที่เรียนมาแล้ว ใน downstream task มักเปลี่ยนหัวให้มีจำนวน output ตาม label ใหม่ ภาพ Visual Feature Hierarchy วาง Feature ทั่วไปในชั้นต้นและตัวแทนเฉพาะงานมากขึ้นในชั้นท้าย Linear Probing ตรึง Backbone แล้วเรียนหัวใหม่ ส่วน Fine-tuning เปิดบางชั้นหรือทั้งหมดให้ปรับ Weight Chain-Thaw เปิดทีละส่วน และ Surgical Fine-tuning ทดลองว่าการเปลี่ยน distribution ส่งผลที่ชั้นใด

ใน Lab ต้องใช้ preprocessing ให้เข้ากับ Backbone เดิม โดยเฉพาะ normalization และขนาดภาพ Negative Transfer คือการย้ายความรู้ที่ทำให้เป้าหมายแย่ลง ส่วน Catastrophic Forgetting กล่าวถึงความรู้เดิมที่สูญหายเมื่อเรียนงานใหม่ การเลือก freeze จึงผูกกับความคล้ายของข้อมูล เป้าหมาย และต้นทุน

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 29–38

## Intermediate Activations, Saliency และ CAM

ภาพตัวอย่าง CNN ใช้ฉากหลังทำนายคลาส ชี้ว่าคะแนนสูงอาจเกิดจากสิ่งที่ไม่ตรงเจตนาของงาน Intermediate Activations ตอบว่า Filter ตอบสนองที่ไหน Saliency ดูความไวของคะแนนคลาสต่อ Pixel และ CAM รวม Feature Maps ด้วย Weight ของหัวทำนายที่ใช้ Global Average Pooling วิธีเหล่านี้ตอบคำถามต่างกัน การเปรียบเทียบ Heatmap ควรตรวจ target class และชั้นที่เลือกก่อนตีความสี

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 39–49

<a name="grad-cam"></a>

## Grad-CAM

ให้ $A^k$ เป็น Feature Map ช่อง $k$ และ $s_c$ เป็นคะแนนคลาส $c$ ก่อน Softmax คำนวณ Gradient ของ $s_c$ ต่อแต่ละตำแหน่ง $(i,j)$ ใน $A^k$ แล้วเฉลี่ย spatial เป็น $\alpha_k^c$ จากนั้นได้

$$
H_c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right).
$$

Gradient ที่เฉลี่ยบอกน้ำหนักของช่องสำหรับคลาสที่เลือก Feature Map บอกตำแหน่งในภาพ ส่วน ReLU เก็บสัญญาณบวกตามนิยามในสไลด์ Heatmap ความละเอียดต่ำอาจแยกหลายวัตถุคลาสเดียวกันได้ยาก และเป็นหลักฐานสำหรับ Debugging ภาพบริเวณสว่างเพียงอย่างเดียวยืนยันสาเหตุของการทำนายไม่ได้

ภาพ Visualizing Filters ใช้การปรับ input เพื่อดูสิ่งที่กระตุ้น Feature Map และอาจได้รูปแบบที่ไม่เหมือนภาพธรรมชาติ Lab Grad-CAM อธิบาย Hook และ FX สำหรับดึง Activation ถ้าปิด Gradient Tracking ใน Forward ที่ต้องใช้ Gradient จะคำนวณ $\alpha_k^c$ ไม่ได้

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 49–57

## ข้อจำกัดด้านหน่วยความจำและ Ablation

Training เก็บ Activation สำหรับ Backpropagation นอกเหนือจาก Weight, Gradient และ Optimizer State จึงใช้หน่วยความจำคนละรูปแบบกับ Inference สไลด์ยก batch size, precision และขนาดภาพเป็นปัจจัยที่ควบคุมได้ ส่วน Ablation เปลี่ยนหรือเอาองค์ประกอบหนึ่งออกแล้วประเมินซ้ำด้วยเงื่อนไขที่เทียบกันได้ เพื่อศึกษาว่าส่วนนั้นเกี่ยวข้องกับพฤติกรรมใดของระบบ

> **อ้างอิง** [class05/5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 58–61

## ทบทวนและเชื่อมบท

เมื่อเลือก CNN ให้ระบุทั้งโจทย์ Metric และฮาร์ดแวร์ เมื่อย้าย Backbone ให้บันทึก preprocessing และชั้นที่ freeze เมื่ออธิบาย prediction ให้แยกการหาตำแหน่งที่โมเดลใช้จากการตรวจว่าคำทำนายถูกต้อง

## คำถามทบทวน

1. Adaptive Pooling กับ normalization ของ pretrained CNN แก้ปัญหาคนละขั้นอย่างไร
2. เหตุใด Grad-CAM ใช้ eval mode พร้อม Gradient Tracking

[คำถามและเฉลย](../review/practice-and-solutions.md#class05) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| จาก LeNet และ AlexNet สู่ Data Augmentation | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 2–14 |
| VGG, Inception, ResNet และตระกูลอื่น | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 15–28 |
| Transfer Learning และการเลือกชั้น | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 29–38 |
| Intermediate Activations, Saliency และ CAM | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 39–49 |
| Grad-CAM | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 49–57 |
| ข้อจำกัดด้านหน่วยความจำและ Ablation | [5_CNN2_classifier_2handout_color.pdf](https://drive.google.com/file/d/1yhDX_PxzPKyDOjFM1xSsnVY2pcCaOz8F/view) · PDF หน้า 58–61 |

[← ก่อนหน้า](04-cnn-basics.md) | [Index](../INDEX.md) | [ถัดไป →](06-rnn.md)
