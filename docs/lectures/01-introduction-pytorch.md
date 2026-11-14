# 1. Introduction และ PyTorch

[Index](../INDEX.md) | [ถัดไป →](02-mlp-basics.md)

## แหล่งเรียน

class01 · ตรวจต้นฉบับ 2026-11-14

- [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view)
- Lab [setup_pytorch](../labs/setup_pytorch.md)

## เป้าหมายและพื้นฐาน

เมื่ออ่านจบควรอธิบายได้ว่า Deep Learning อยู่ตรงไหนใน AI, เหตุใดการฝึกจึงเกี่ยวกับข้อมูลและฮาร์ดแวร์ และอ่าน shape ของ Tensor ใน PyTorch ได้ เตรียมพื้นฐาน Python และ NumPy ตามภาพใน PDF หน้า 5 แล้วใช้ Lab ตรวจ environment ของตัวเอง

## ขอบเขตและภาพรวมของคอร์ส

เอกสารเปิดด้วยโครงวิชา สัดส่วนกิจกรรมในชั้น งานเขียนโปรแกรม และโครงงาน รวมถึงสื่อที่แนะนำ ภาพโจทย์ Sudoku ช่วยแยกขั้นตอนให้เห็นว่าในระบบจริง อาจต้องเตรียมภาพและหาตำแหน่งช่องก่อนส่งข้อมูลเข้าโมเดล จากนั้นจึงนำผลทำนายไปใช้แก้ปัญหาต่อ ผู้อ่านจึงระบุได้ว่าโมเดลรับอะไรและผลที่ส่งออกไปยังขั้นต่อไปคืออะไร

ภาพ FAQ วาง Python และ NumPy ไว้ใกล้พื้นฐาน ML ส่วน Deep Reinforcement Learning, Geometric Deep Learning และ agent harness อยู่ในสไลด์ขอบเขตที่ไม่ได้สอน รายละเอียดการประเมินให้ยึด handout ของรุ่นเรียน

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 1–8

## จาก Neural Network ไปสู่ Deep Learning สมัยใหม่

ภาพซ้อนกลุ่มในหน้า 9 วาง Deep Learning ภายใน Machine Learning และ Machine Learning ภายใน AI สไลด์ใช้ Universal Approximation Theorem เพื่อชี้ความสามารถในการประมาณฟังก์ชันของ Feed-forward Network ที่มี Hidden Layer เงื่อนไขเชิงทฤษฎีและข้อพิสูจน์เต็มยังยืนยันจากสไลด์นี้ไม่ได้ ความสามารถในการแทนฟังก์ชันกับการหาพารามิเตอร์ที่เหมาะสมระหว่างฝึกเป็นคนละคำถาม

Timeline เชื่อม Backpropagation, CNN และ LSTM ไปถึง AlexNet, Transformer, BERT และ Generative AI กราฟประกอบใช้คนละตัววัด เช่น Benchmark, จำนวนโมเดลเด่น, การเข้าถึงโมเดล, การเผยแพร่ Dataset และการเข้าร่วมงานประชุม จึงควรอ่านชื่อแกนและช่วงเวลาแยกกัน ภาพโมเดลในวิชาวาง MLP, CNN และกลุ่มโมเดลลำดับไว้ข้างกลุ่ม Generative Models

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 9–18

## Feature Extraction และต้นทุนของการฝึก

แผนภาพเริ่มจากภาพรถ ในเส้นทาง ML ที่ยกมา ผู้พัฒนาเตรียม Features ก่อนส่งให้ Classifier ส่วนเส้นทาง Deep Learning วาง Feature Extraction กับ Classification ในโครงข่ายเดียวกัน พารามิเตอร์ในชั้นต้นจึงเกี่ยวข้องกับสิ่งที่ชั้นตัดสินใจใช้ด้วย ข้อมูล ปริมาณ Compute และ Algorithm เป็นปัจจัยร่วมในภาพอธิบายการเติบโต

ต้นฉบับยกต้นทุนการรวบรวมและติดป้ายข้อมูล การฝึกและ Inference รวมถึงความยากของการอธิบายการตัดสินใจไว้คู่กับประโยชน์ ให้แยกคำถามว่าโมเดลทำงานได้ดีตามตัววัดใดจากคำถามว่าเรารู้เหตุผลของผลทำนายแต่ละกรณีเพียงใด

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 18–20

## Neural Scaling Laws

สไลด์ใช้กราฟ log–log แสดงความสัมพันธ์ระหว่าง Loss กับขนาดโมเดล ข้อมูล และ Compute แนวคิด Kaplan ใน handout เน้นแนวโน้มที่คาดประมาณได้เมื่อเพิ่มทรัพยากร ขณะที่ Chinchilla ตั้งคำถามว่าควรแบ่งงบ Compute ระหว่างจำนวนพารามิเตอร์กับจำนวน Tokens อย่างไร ภาพเปรียบเทียบจึงควรอ่านจากทรัพยากรที่กำลังเปลี่ยนและเงื่อนไขที่ใช้สร้างกราฟ

ภาพความสามารถที่ปรากฏหลังผ่านระดับ Scale วางกราฟความสามารถรายงานไว้ข้างเส้น Loss ที่เปลี่ยนค่อนข้างเรียบ ทั้งสองแกนวัดคนละสิ่ง กราฟจุดช่วงท้ายแสดงแนวโน้มขนาดโมเดล ขนาดชุดข้อมูล และเวลาฝึกตามปี

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 21–25

## Hardware Acceleration

ภาพ Local Machine เทียบกับ Cloud ช่วยแยกเครื่องที่จัดการเองจาก Runtime ที่ผู้ให้บริการจัดให้ บทเรียนใช้ Colab เป็นทางเลือกหนึ่งพร้อมยกข้อจำกัดของระยะเวลาและทรัพยากร ภาพ CPU, GPU และ TPU พูดถึงรูปแบบงานและ Throughput ตามโครงข่ายและขนาด Batch ส่วน NPU และ LPU เป็นตัวอย่างการออกแบบให้เหมาะกับงานเฉพาะ ภาพ SoC วาง NPU, CPU และ GPU ไว้ร่วมกันเพื่อให้เห็นบทบาทที่ประกอบกัน

ช่วงท้ายเชื่อมชิปกับระบบขนาดใหญ่ผ่านกรณี wafer-scale, ศูนย์ข้อมูล, กำลังไฟ และการใช้น้ำ กราฟกับภาพข่าวเป็นภาพตามเวลาที่ระบุใน handout จึงใช้อ่านชนิดของต้นทุนและหน่วยวัด

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 26–37

## Framework และ Computation Graph

Framework ช่วยให้เขียนการคำนวณแบบ Tensor และสร้าง Neural Network ภาพ Static Computation Graph แยกขั้นประกาศโครงออกจากขั้นทำงาน ส่วน Dynamic Computation Graph ประกอบกราฟตามการคำนวณที่เกิดขึ้นขณะรัน ความต่างนี้ปรากฏเมื่อโปรแกรมปรับลำดับการคำนวณตามข้อมูล

เอกสารเปรียบเทียบ TensorFlow, PyTorch และ Keras ผ่านตัวอย่างโค้ด การสำรวจ และภาพ Keras หลาย Backend ตัววัดจากการสำรวจผู้ใช้ บทความ และโครงการซอฟต์แวร์ตอบคนละคำถาม Lab ใช้ Python และ PyTorch ให้เริ่มจากตรวจ import และ device ใน [setup_pytorch](../labs/setup_pytorch.md)

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 38–44

<a name="tensor"></a>

## Tensor, FLOPS และการนำโมเดลไปใช้

Tensor เป็นคำรวมสำหรับข้อมูลหลายมิติ Scalar ใช้ศูนย์ดัชนี Vector ใช้หนึ่งดัชนี Matrix ใช้สองดัชนี ใน Lab เราอ่าน shape (5, 3, 4) เป็นห้าตัวอย่าง ตัวอย่างละสามแถวสี่คอลัมน์ การระบุความหมายของแต่ละแกนช่วยให้ตรวจว่าโค้ดใช้ Batch หรือแกนภายในตัวอย่าง

FLOPS วัด Floating-point Operations ต่อวินาที ส่วน FLOPs วัดปริมาณ Operations ที่นับได้ ภาพ ONNX วางรูปแบบแลกเปลี่ยนโมเดลและ ONNX Runtime ไว้ระหว่างเครื่องมือฝึกกับเป้าหมาย Deployment ภาพ Carbon Emissions เชื่อมผลกระทบของ Compute ทั้งฝึกและให้บริการ หน้า 49 ให้เตรียมเครื่องหรือ Cloud สำหรับคลาสถัดไป

> **อ้างอิง** [class01/1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 45–50

## ทบทวนและเชื่อมบท

เมื่ออ่านกราฟ Compute ให้ตรวจหน่วยและสิ่งที่นับ เมื่ออ่าน Tensor ให้ตรวจชื่อและความยาวทุกแกน ก่อนเข้าบท MLP ลองตอบว่าข้อมูลหนึ่งตัวอย่างกับหนึ่ง Batch จะมี shape ต่างกันอย่างไร และ PyTorch รายงานการรองรับ GPU ตรงกับที่เครื่องมองเห็นหรือไม่

## คำถามทบทวน

1. อ่าน shape (5,3,4) แล้วบอกว่าแต่ละแกนคืออะไร
2. ความพร้อมของเครื่องมือ NVIDIA และของ PyTorch ต่างกันอย่างไร

[คำถามและเฉลย](../review/practice-and-solutions.md#class01) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| ขอบเขตและภาพรวมของคอร์ส | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 1–8 |
| จาก Neural Network ไปสู่ Deep Learning สมัยใหม่ | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 9–18 |
| Feature Extraction และต้นทุนของการฝึก | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 18–20 |
| Neural Scaling Laws | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 21–25 |
| Hardware Acceleration | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 26–37 |
| Framework และ Computation Graph | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 38–44 |
| Tensor, FLOPS และการนำโมเดลไปใช้ | [1_Intro_2handout_color.pdf](https://drive.google.com/file/d/1tX0wztQGILPBKlJ7YSMVyEfjfaW5SqPN/view) · PDF หน้า 45–50 |

[Index](../INDEX.md) | [ถัดไป →](02-mlp-basics.md)
