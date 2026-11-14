# 6. Recurrent Neural Network

[← ก่อนหน้า](05-cnn-classifier.md) | [Index](../INDEX.md) | [ถัดไป →](07-lstm-gru.md)

## แหล่งเรียน

class06 · ตรวจต้นฉบับ 2026-11-14

- [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view)
- Lab [RNN_ex1_ex2_ex3](../labs/RNN_ex1_ex2_ex3.md)
- Lab [RNN_ex4_ex5](../labs/RNN_ex4_ex5.md)
- Lab [RNN_ex6](../labs/RNN_ex6.md)

## เป้าหมายและพื้นฐาน

อ่านแผนภาพ unroll ของ RNN, ติดตาม Hidden State และจัด Window สำหรับ Time Series ได้ ทวน Chain Rule และ Mini-batch จาก [Class 02](02-mlp-basics.md) ก่อนแยกแกนเวลาออกจากแกน Batch

## ชนิด Input และ Output ตามเวลา

ภาพใน handout แยก 1→1 แบบจำแนกภาพ, 1→N เช่น Image Captioning, N→1 เช่น Sentiment Classification, N→N แบบตำแหน่งตรงกัน เช่น ป้ายต่อ frame และ M→N ที่ความยาวต่างกัน เช่น Translation เมื่อเริ่มออกแบบให้ระบุว่างานต้องการผลเมื่อจบ input หรือทุก timestep กรณีข้อความ ประโยคที่มีทั้งคำเชิงบวกและลบแสดงว่าความหมายขึ้นกับบริบทรอบคำและลำดับ

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 1–8

<a name="recurrent-state"></a>

## Vanilla RNN และการ Unroll

ภาพ loop เมื่อคลี่ตามเวลาจะเห็น Cell ที่ใช้ $W_x,W_h$ ชุดเดียวกันในทุก timestep สำหรับ feature vector $x_t\in\mathbb R^k$ และ Hidden State $h_{t-1}\in\mathbb R^d$ เขียนแบบ Bias รวมได้ว่า

$$
a_t=W_xx_t+W_hh_{t-1}+b,\qquad h_t=\tanh(a_t).
$$

$W_x$ มีขนาด $d\times k$, $W_h$ ขนาด $d\times d$ และ $b$ มี $d$ ค่า PyTorch แยก input bias และ recurrent bias ซึ่งบวกเข้าด้วยกันใน preactivation หัวทำนาย $W_yh_t+b_y$ แปลง hidden features เป็นขนาดที่โจทย์ต้องการ สไลด์ตัวอักษร dogs ใช้ one-hot ขนาด 4, Hidden State ขนาด 3 และเริ่ม $h_0$ เป็นศูนย์ ใน step แรกส่วน $W_hh_0$ จึงเป็นศูนย์ Step ต่อมาส่วนนี้นำผลของอดีตเข้ามา

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 9–13

## Backpropagation Through Time

ในภาพ BPTT Loss อาจรวมจาก output หลายเวลา แต่ Weight ชุดเดียวปรากฏทุกครั้งในกราฟที่ unroll อนุพันธ์ของ Weight จึงต้องรวมผลจากหลายตำแหน่ง เช่น Loss เวลา 3 ผ่าน state เวลา 3 แล้วไหลย้อนต่อไปถึงเวลา 2 และ 1 ได้ การคูณอนุพันธ์หลายช่วงทำให้ขนาด Gradient อ่อนไหวกับความยาวของเส้นทาง ซึ่งเป็นโจทย์ของ Class 07 สไลด์ Regularization แยก Dropout บนเส้นเชื่อมระหว่าง recurrent layers ออกจากการตัดสัญญาณที่ส่งข้ามเวลา

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 14–15

## Batch, Window และการป้องกัน Leakage

เมื่อ batch_first=True input เป็น (B,T,k) และ output ของ recurrent layer เป็น (B,T,d) ส่วน final hidden state จัดตาม layer/direction และ Batch หากต้องการทำนายจากเวลาเดียวจึงเลือก out[:, -1, :] ได้เมื่อหน้าต่างแต่ละตัวอย่างยาวเท่ากัน

ภาพ Time Series ใช้ Window ของข้อมูลก่อนหน้าเป็น input และค่าภายหลังเป็น target แบ่ง train, validation และ test ตาม timestamp ก่อนสร้าง Windows เพื่อให้ชุดฝึกไม่เห็นอนาคต ถ้าทำ normalization ให้ fit จากชุดฝึกแล้วใช้ค่านั้นกับชุดอื่นและกลับสเกลด้วยสถิติเดิม การสลับหน้าต่างภายใน train เป็นอีกขั้นหนึ่งหลังแยกข้อมูลแล้ว

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 16–19

## Stacked RNN และ Multi-step Forecasting

Stacked RNN ส่ง state ไปยัง timestep ถัดไปในชั้นเดียวกัน และส่ง output ขึ้นชั้นถัดไปที่เวลาเดียวกัน Lab แรกเทียบ Window กับจำนวน Layers และใช้โมเดลทำนายทีละจุดซ้ำโดยเลื่อน Window แล้วป้อนผลที่เพิ่งได้กลับเข้าไป Error จึงส่งต่อได้ สไลด์ยังแสดงการทำนายเวกเตอร์อนาคตจาก timestep สุดท้าย และการทำนายหลาย horizon ที่ทุก timestep โดยแบบหลังได้รับ Loss จากหลายจุดใน Window

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ถ้า Window ยาว 3 เริ่มจาก [3, 6, 9] แล้วโมเดลทำนาย 11 ขั้น recursive ใช้ [6, 9, 11] ต่อ แม้ค่าจริงของจุดนั้นเป็น 12 กรณีวัดแบบ one-step จากข้อมูลจริงจะใช้ [6, 9, 12] จึงกำลังวัดคนละพฤติกรรม

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 20–23

## Encoder–Decoder และ Teacher Forcing

Encoder สรุป Window เป็น final Hidden State ให้ Decoder ภาพการฝึกใช้ target ของ step ก่อนเป็น input ของ step ต่อไปเพื่อให้ Decoder ได้บริบทที่ถูก ส่วน Inference ต้องใช้ prediction เพราะอนาคตยังไม่ทราบ ความต่างนี้เรียก Exposure Bias ในบทเรียน Scheduled Sampling จึงค่อยสลับแหล่งของ input ระหว่าง ground truth กับ prediction ตามระยะฝึก Lab EX6 ใช้ค่า observed ตัวสุดท้ายเปิด Decoder และเก็บ output ต่อกันตามแกนเวลา

> **อ้างอิง** [class06/6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 24–28

## ทบทวนและเชื่อมบท

คำถามที่ควรถามก่อนอ่านกราฟคือหัวทำนายสร้างกี่ค่าและ ณ เวลาใด แล้วช่วงทดสอบใช้ค่าจริงหรือผลทำนายเป็น input [Class 07](07-lstm-gru.md) เพิ่มกลไกที่ช่วยจัดการบริบทข้ามเวลายาวขึ้น

## คำถามทบทวน

1. ไล่ shape ของ RNN ที่รับ (16,50,1) และมี hidden size 64
2. วาดสอง Window ถัดไปสำหรับ one-step และ recursive เมื่อ prediction แรกคลาดจากค่าจริง

[คำถามและเฉลย](../review/practice-and-solutions.md#class06) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| ชนิด Input และ Output ตามเวลา | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 1–8 |
| Vanilla RNN และการ Unroll | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 9–13 |
| Backpropagation Through Time | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 14–15 |
| Batch, Window และการป้องกัน Leakage | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 16–19 |
| Stacked RNN และ Multi-step Forecasting | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 20–23 |
| Encoder–Decoder และ Teacher Forcing | [6_RNN_2handout_color.pdf](https://drive.google.com/file/d/1BGx10e31bXXg5GAulqonQTTxltF73R0F/view) · PDF หน้า 24–28 |

[← ก่อนหน้า](05-cnn-classifier.md) | [Index](../INDEX.md) | [ถัดไป →](07-lstm-gru.md)
