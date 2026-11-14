# 4. Convolution และ CNN

[← ก่อนหน้า](03-mlp-problems-visualization.md) | [Index](../INDEX.md) | [ถัดไป →](05-cnn-classifier.md)

## แหล่งเรียน

class04 · ตรวจต้นฉบับ 2026-11-14

- [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view)
- Lab [CNN1_ex2](../labs/CNN1_ex2.md)
- Lab [CNN1_ex3](../labs/CNN1_ex3.md)
- Lab [CNN1_ex4](../labs/CNN1_ex4.md)

## เป้าหมายและพื้นฐาน

เรียนรู้วิธีตาม shape และนับพารามิเตอร์ของ CNN ตั้งแต่ Kernel ไปถึง Classifier ใช้ [Class 02](02-mlp-basics.md) ทวน Linear และ Loss และ [Class 03](03-mlp-problems-visualization.md) ทวน Batch Normalization

## จาก ImageNet สู่การแทนภาพ

Handout ใช้ ImageNet และผลของ AlexNet เป็นจุดเชื่อมจากประวัติ CNN สู่ Image Classification ถ้า Flatten ภาพแล้วต่อทุก Pixel เข้ากับทุก Neuron จำนวน Weight โตตามจำนวน Pixel และตำแหน่งที่ภาพเลื่อนไปมี Weight คนละชุด CNN ใช้ Kernel เดิมเลื่อนไปตามตำแหน่ง จึงเรียนตัวตรวจรูปแบบท้องถิ่นที่ใช้ซ้ำได้

ตัวอย่างภาพ RGB ห้าภาพในสไลด์มีรูป (5, 3, 400, 600) แบบ NCHW สำหรับ PyTorch Conv2d เทียบกับ (5, 400, 600, 3) แบบ NHWC ให้อ่านก่อนว่า Tensor เรียงแกนอย่างไร ส่วน channels_last ใน PyTorch เป็นเรื่อง Memory Format ซึ่งสไลด์แยกจากการสลับความหมายของแกน

> **อ้างอิง** [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 1–7

<a name="spatial-convolution"></a>

## Spatial Convolution

ภาพในหน้า 8 วาง Kernel ทับหน้าต่างของ input แล้วคูณสมาชิกตำแหน่งเดียวกัน จากนั้นบวกทั้งหมดเป็นค่าหนึ่งช่องของ Feature Map การเลื่อนด้วย Stride กำหนดตำแหน่งที่คำนวณ ส่วน Padding เพิ่มพื้นที่รอบขอบเพื่อควบคุมการครอบคลุมและขนาดผลลัพธ์ สำหรับหนึ่งแกนที่มีขนาด $H$, Kernel $K$, Padding ข้างละ $P$, Stride $S$ และ Dilation $D$ การนับตำแหน่งที่ Kernel วางได้ให้

$$
H_\text{out}=\left\lfloor\frac{H+2P-D(K-1)-1}{S}+1\right\rfloor.
$$

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เมื่อ $H=5,K=3,P=1,S=2,D=1$ หน้าต่างเลื่อนได้สามตำแหน่ง หากลด Padding เป็น 0 เหลือสองตำแหน่ง สำหรับภาพคำนวณซ้ำอีกครั้งตามแกนกว้าง

> **อ้างอิง** [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 8–11, 15

## Channels, Groups และ Dilation

Conv2d หนึ่ง output channel ใช้ Kernel 2D หนึ่งชิ้นต่อ input channel รวม Feature Map ของทุก input channel ที่เชื่อมกัน แล้วบวก Bias ของ output channel นั้น ดังนั้น filter สี่ชุดให้ output สี่ channel แม้ input จะเป็น RGB สาม channel ในภาพ Grouped Convolution เส้นเชื่อมระหว่างบางกลุ่มถูกตัดออก โดย in_channels และ out_channels ต้องหารด้วย groups ลงตัว กรณี groups เท่า in_channels เป็นพื้นฐานของ Depthwise Convolution

Dilation เว้นช่องระหว่างจุดของ Kernel ทำให้หน้าต่างมองพื้นที่กว้างขึ้นโดยคงจำนวน Weight ต่อ filter ภาพ Conv1d, Conv2d และ Conv3d ชวนดูจำนวนแกนที่ Kernel เลื่อนไป ไม่ใช่นับจำนวน channels เพื่อเลือกชั้น

> **อ้างอิง** [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 10–13

## Pooling, Strided Conv และ Spatial BatchNorm

Max Pooling เก็บค่าสูงสุดในแต่ละหน้าต่าง ส่วน Average Pooling เก็บค่าเฉลี่ย ภาพ downsampling แสดงว่า spatial size ลดลงโดยยังมีจำนวน channels เดิม การทิ้งรายละเอียดช่วยลดการคำนวณ แต่หากงานต้องระบุตำแหน่งละเอียด เช่น Segmentation ต้องพิจารณาผลของการสูญเสียตำแหน่ง Strided Conv ลด spatial size พร้อม Weight ที่เรียนรู้ได้ ส่วน Pooling ใช้กฎตายตัว

BatchNorm2d ใช้สถิติของแต่ละ channel ข้าม Batch และตำแหน่ง spatial ตามภาพหน้า 16 เมื่อตาม shape ของ Conv → Activation → Pool → Flatten ให้แยกช่วงที่ขนาดเปลี่ยนจากช่วงที่เพียงปรับค่า

> **อ้างอิง** [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 13–16

## ประกอบ Feature Extractor และ Classifier

ภาพ CNN สำหรับจำแนกภาพไล่จาก Feature Maps ขนาดใหญ่ไปยังตัวแทนที่เล็กลง ก่อนส่งให้ Classifier ใน [CNN1_ex2](../labs/CNN1_ex2.md) เส้นทางตรงสร้างด้วย Sequential หรือ Subclassing ก็ได้ ส่วน [CNN1_ex3](../labs/CNN1_ex3.md) ต้องจัดทางแยกของสอง inputs และสอง outputs ใน forward ด้วยตัวเอง [CNN1_ex4](../labs/CNN1_ex4.md) ใช้ MNIST เปรียบเทียบ MLP ที่ Flatten ภาพตั้งแต่ต้นกับ CNN ที่รักษา spatial axes ระหว่างสกัด Features

> **อ้างอิง** [class04/4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 17–21

## ทบทวนและเชื่อมบท

ลองนับขนาดทีละแกนและจำนวน Weight ต่อ filter ก่อนรวม Bias จากนั้นตรวจว่า Pooling ลดอะไรและรักษาอะไร [Class 05](05-cnn-classifier.md) ใช้บล็อกเหล่านี้เพื่ออธิบายสถาปัตยกรรม ImageNet และการนำ Weight เดิมไปใช้

## คำถามทบทวน

1. นับพารามิเตอร์ Conv2d ที่มี 3 input channels, 16 output channels และ Kernel 3×3
2. ไล่ขนาดภาพ 28×28 ผ่าน Conv ที่รักษาขนาดแล้วตามด้วย MaxPool 2×2

[คำถามและเฉลย](../review/practice-and-solutions.md#class04) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| จาก ImageNet สู่การแทนภาพ | [4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 1–7 |
| Spatial Convolution | [4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 8–11, 15 |
| Channels, Groups และ Dilation | [4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 10–13 |
| Pooling, Strided Conv และ Spatial BatchNorm | [4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 13–16 |
| ประกอบ Feature Extractor และ Classifier | [4_CNN1_basic_2handout_color.pdf](https://drive.google.com/file/d/1l_i3mn4KHUidvTlocn4-iWQJ8nKw7B26/view) · PDF หน้า 17–21 |

[← ก่อนหน้า](03-mlp-problems-visualization.md) | [Index](../INDEX.md) | [ถัดไป →](05-cnn-classifier.md)
