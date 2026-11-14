# 3. ปัญหาของ MLP และการติดตามการทดลอง

[← ก่อนหน้า](02-mlp-basics.md) | [Index](../INDEX.md) | [ถัดไป →](04-cnn-basics.md)

## แหล่งเรียน

class03 · ตรวจต้นฉบับ 2026-11-14

- [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view)
- Lab [MLP2_ex1-1_StaticViz](../labs/MLP2_ex1-1_StaticViz.md)
- Lab [MLP2_ex1-2_TensorBoard](../labs/MLP2_ex1-2_TensorBoard.md)
- Lab [MLP2_ex1-3_WandB](../labs/MLP2_ex1-3_WandB.md)

## เป้าหมายและพื้นฐาน

อ่าน Gradient และ Learning Curves เพื่อแยกปัญหาการ Optimize ออกจาก Generalization และติดตามการเลือก Checkpoint ได้ ควรเข้าใจ Forward Pass, Loss, Optimizer และ model mode จาก [Class 02](02-mlp-basics.md)

## Vanishing และ Exploding Gradients

ภาพลูกศรย้อนจาก Layer ท้ายไปต้นใน handout แสดง Gradient ที่ต้องผ่านอนุพันธ์หลายชั้น หากตัวคูณมีขนาดต่ำต่อเนื่อง สัญญาณที่กลับไปถึงชั้นต้นจะเล็กจน Weight เปลี่ยนน้อย ถ้าขนาดเพิ่มต่อเนื่อง การอัปเดตอาจแกว่งหรือเกิดค่าผิดปกติ การดู Gradient ต่อ Layer จึงตอบคำถามที่ Loss รวมทั้งโมเดลตอบไม่ได้

ReLU ไม่อิ่มตัวฝั่งบวก ส่วนการเลือก Initialization คุมขนาดสัญญาณตั้งแต่เริ่ม Xavier ใช้ทั้ง fan-in และ fan-out เพื่อหาสเกลร่วม ตารางในหน้า 5 เปรียบเทียบ Glorot, He และ LeCun โดยเชื่อมกับ Activation กราฟกระจาย Activation ข้าม Layer ช่วยตรวจว่าค่าหดหรือกระจายออกมากเพียงใด

ให้ $n_\mathrm{in}$ เป็นจำนวนสัญญาณเข้าต่อ unit และ $n_\mathrm{out}$ เป็นจำนวนสัญญาณที่ส่งต่อจากแต่ละ input สำหรับ Xavier แบบ uniform ในสไลด์ สุ่ม Weight ระหว่าง $-a$ และ $a$ เมื่อ $a=\sqrt{6/(n_\mathrm{in}+n_\mathrm{out})}$ ดังนั้น Layer ที่มีจำนวนทางเชื่อมมากขึ้นจะได้ช่วงการสุ่มแคบลงตามสูตร ส่วน He ในตารางใช้ variance ตาม fan-in และวางคู่กับ ReLU วิธีเลือกจึงขึ้นกับทั้งจำนวนการเชื่อมและ Activation ที่จะรับค่าจาก Weight เหล่านั้น

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 2–5

<a name="batch-normalization"></a>

## Batch Normalization

สำหรับ feature หนึ่งและตัวอย่าง $i$ ใน Mini-batch $B$ คำนวณ mean $\mu_B$ และ variance $\sigma_B^2$ แล้วปรับด้วย $\hat x_i=(x_i-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ ก่อนใช้ $y_i=\gamma\hat x_i+\beta$ โดย $\epsilon$ ช่วยเมื่อ variance เล็ก และ $\gamma,\beta$ เป็นพารามิเตอร์เรียนรู้ได้ ภาพกล่องพารามิเตอร์ยังมี running mean และ running variance ซึ่งสะสมจากการฝึกและนำไปใช้ใน Inference

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** ถ้า feature ใน Batch มีค่า 2 และ 6 ค่าเฉลี่ยคือ 4, variance คือ 4 และค่ามาตรฐานประมาณ −1 กับ 1 เมื่อ $\epsilon$ เล็ก ถ้า $\gamma=2,\beta=1$ ผลหลัง scale และ shift คือประมาณ −1 กับ 3 ข้อมูลสองจุดนี้อธิบายการคำนวณใน Training ส่วน Inference อ่านสถิติสะสม

สไลด์แสดงทั้งตำแหน่งก่อนและหลัง Activation และยกข้อจำกัดด้าน batch size, ความต่างระหว่าง train กับ eval และการใช้งานแบบกระจาย ใน Lab วาง BatchNorm1d หลัง Linear และก่อน ReLU

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 6–9

> **อ้างอิง** [class03/code/MLP2_ex1-1_StaticViz.ipynb](https://drive.google.com/file/d/14hvRoGYzQ9DRh2IFoNODLto5NuSGobRL/view) · cells 18, 27–29

## Overfitting และ Regularization

ภาพ Underfit, Optimal และ Overfit ใช้จุดชุดเดียวกันเพื่อแสดงว่าเส้นที่โค้งรับทุกจุดอาจตอบสนอง Noise มากเกินไป เทียบ train กับ validation และพิจารณาข้อจำกัดของข้อมูลก่อนตีความว่า error สูงมาจาก Bias หรือ Variance Early Stopping ใช้ผล validation เลือกจุดหยุดหรือ snapshot ส่วน L1 และ L2 เพิ่ม penalty ตาม $L_\text{total}=L+\lambda R(w)$ โดยใช้ $R(w)=\sum_j|w_j|$ และ $R(w)=\sum_jw_j^2$ ตามลำดับ $\lambda$ คุมแรงของ penalty ภาพและข้อความเปรียบเทียบ L1 ที่ให้ Weight บางตัวเป็นศูนย์กับ L2 ที่ดัน Weight ให้เล็กลง

Dropout สุ่มปิดบาง Activation ระหว่างฝึก ภาพสอง Subnetworks แสดงว่าเส้นทางที่มีผลเปลี่ยนตาม mask ใน PyTorch การสเกลค่าที่รอดด้วย $1/(1-p)$ ระหว่างฝึกชดเชยค่าเฉลี่ยที่ลดลง การประเมินใน eval mode ปิด Dropout จึงเกิดกรณีที่ validation loss ต่ำกว่า training loss ได้แม้ใช้ Loss เดียวกัน เพราะโมเดลทำงานต่างโหมด หากต้องการวินิจฉัยให้เปรียบเทียบกับผลของชุดฝึกที่วัดใน eval mode ด้วย

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 10–19

## Uncertainty และความน่าเชื่อถือ

MC Dropout เปิดเฉพาะชั้น Dropout อีกครั้งใน Inference แล้วอ่านการกระจายจากหลาย Forward Pass ของตัวอย่างเดิม ภาพจุดทำนายที่กระจุกหรือกระจายช่วยอธิบายเหตุที่ prediction เดียวให้ข้อมูลไม่ครบ ค่าใช้จ่ายเพิ่มตามจำนวนครั้ง และวิธีนี้อาศัยโมเดลที่มี Dropout อยู่แล้ว Aleatoric Uncertainty เกี่ยวกับ Noise ของข้อมูล ส่วน Epistemic Uncertainty เกี่ยวกับความรู้ที่โมเดลยังมีจำกัด

ตารางใน handout แยก Predictive Performance, Calibration และ Predictive Uncertainty เช่น Accuracy วัดความถูกต้องรวม ขณะที่ Calibration ดูว่ากลุ่มที่ให้ความมั่นใจประมาณ 0.8 ถูกจริงประมาณ 80% หรือไม่ NLL และ ECE เป็นตัวอย่างตัววัดที่ดูคุณภาพความน่าจะเป็น Multiple Seeds ตรวจความไวต่อการสุ่ม ส่วน K-fold เปลี่ยนชุด validation และตอบคำถามเรื่องความไวต่อการแบ่งข้อมูล

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 20–26

## ติดตามการทดลองและเลือก Checkpoint

ทั้งสาม Lab ใช้ Regression บน Boston Housing และเก็บโมเดลตาม validation MSE ต่ำสุด StaticViz วาดกราฟหลังฝึก TensorBoard เขียน event files เพื่ออ่านตาม Epoch และ W&B เพิ่มการเก็บ config กับ Sweep ให้เปรียบเทียบหลาย run เมื่อปรับ Hyperparameters ซ้ำโดยดูชุด validation เดิมมาก การเลือกอาจเข้ากับ split นั้นเป็นพิเศษ สไลด์ยก K-fold และ nested CV เป็นทางเลือกโดยต้องแลกกับต้นทุนฝึกเพิ่ม

เวลาอ่านกราฟ ให้หาค่าต่ำสุดของ validation loss ก่อนดู Epoch สุดท้าย จากนั้นตรวจ Test ของโมเดลที่เลือกแล้วเพียงในบทบาทประเมินผล กรณี Tabular Data ในการบ้านให้ตั้งการเปรียบเทียบ MLP กับ Traditional ML บนโจทย์เดียวกัน

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 26–33

## Kolmogorov-Arnold Network

ภาพเปรียบเทียบ KAN วางฟังก์ชันหนึ่งตัวแปรที่เรียนรู้ได้ไว้ตามขอบ แล้วให้โหนดรวมสัญญาณ ต่างจากภาพ MLP ที่ขอบใช้ Weight และ Activation อยู่ที่โหนด สไลด์ใช้ B-splines เพื่ออธิบายฟังก์ชันตามช่วง โดยเขียน spline เป็นผลรวมของ basis functions ที่คูณสัมประสิทธิ์ เมื่อเปลี่ยน grid จะเปลี่ยนความละเอียดของฟังก์ชันที่แทนได้ ต้นฉบับยกความสามารถในการอ่านรูปฟังก์ชันและความยืดหยุ่นของ grid ควบคู่กับเวลาและความยากในการฝึก จึงควรประเมินตามชนิดข้อมูลและงบคำนวณ

> **อ้างอิง** [class03/3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 34–39

## ทบทวนและเชื่อมบท

ลองอธิบายว่าทำไม Gradient เล็กตั้งแต่ต้นต่างจาก train loss ที่ลดแต่ validation แย่ลง และทำไม checkpoint จาก validation จึงเหมาะกับการเลือกโมเดลใน Lab บทถัดไปใช้ Spatial Batch Normalization กับข้อมูลภาพ

## คำถามทบทวน

1. BatchNorm ใช้สถิติใดขณะ train และ eval
2. ถ้า validation loss ต่ำกว่า training loss จะตรวจบทบาทของ Dropout อย่างไร

[คำถามและเฉลย](../review/practice-and-solutions.md#class03) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| Vanishing และ Exploding Gradients | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 2–5 |
| Batch Normalization | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 6–9 |
| Overfitting และ Regularization | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 10–19 |
| Uncertainty และความน่าเชื่อถือ | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 20–26 |
| ติดตามการทดลองและเลือก Checkpoint | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 26–33 |
| Kolmogorov-Arnold Network | [3_MLP2_problem_2handout_color.pdf](https://drive.google.com/file/d/179VSRNTM5mCpvWkpvuggpM-YojGadjVZ/view) · PDF หน้า 34–39 |

[← ก่อนหน้า](02-mlp-basics.md) | [Index](../INDEX.md) | [ถัดไป →](04-cnn-basics.md)
