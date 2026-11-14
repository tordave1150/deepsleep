# คำถามทบทวนและเฉลย

[Index](../INDEX.md) · [Formula Reference](formula-reference.md)

คำถามทั้งหมดเป็น **Illustrative Examples (ผู้เรียบเรียงสร้างขึ้น)** โดยใช้แนวคิดและสมการจากบทที่ลิงก์ไว้ ลองตอบก่อนอ่านเฉลย แล้วตรวจว่าระบุ shape, ชุดข้อมูล และเงื่อนไขที่เกี่ยวข้องครบหรือยัง

## class01

[กลับไปบทเรียน](../lectures/01-introduction-pytorch.md)

### พื้นฐาน — Tensor shape (5,3,4) มีตัวอย่างกี่ตัว และการเลือก x[0] เหลืออะไร

**เฉลย** มีห้าตัวอย่าง เลือกตัวแรกแล้วใช้ดัชนีไปหนึ่งแกน เหลือ Matrix (3,4)

### วิเคราะห์ — ถ้า NVIDIA tool รายงาน GPU แต่ PyTorch ไม่ใช้ CUDA ควรดูข้อมูลจากจุดใด

**เฉลย** ดูผล import, build ของ PyTorch และ torch.cuda.is_available() การรายงานของ NVIDIA วัดสิ่งที่ระบบเห็น ส่วน PyTorch วัดสิ่งที่ runtime ใช้ได้

## class02

[กลับไปบทเรียน](../lectures/02-mlp-basics.md)

### คำนวณ — Linear รับ 4 features และออก 3 units ใช้ Weight กับ Bias กี่ค่า

**เฉลย** Weight มี shape (3,4) รวม 12 ค่า Bias มี 3 ค่า รวม 15 ค่า สำหรับ Batch 8 ได้ output (8,3)

### วิเคราะห์ — โมเดลสองตัวทายคลาสถูกเหมือนกัน แต่ให้ probability คลาสจริง 0.6 และ 0.9 ค่า CE ต่างอย่างไร

**เฉลย** สำหรับ one-hot เหลือ −log p ของคลาสจริง ตัวที่ให้ 0.9 จึงได้ Loss ต่ำกว่า แม้การนับถูกผิดจะเท่ากัน

## class03

[กลับไปบทเรียน](../lectures/03-mlp-problems-visualization.md)

### พื้นฐาน — ทำไม BatchNorm ใน eval ใช้สถิติที่สะสม

**เฉลย** ตัวอย่างเดียวหรือ Batch ที่องค์ประกอบต่างไปให้สถิติที่ผันผวน Eval ใช้ running mean/variance ที่เก็บจาก train ขณะที่ scale/shift เป็นค่าที่เรียนมา

### วิเคราะห์ — validation loss ต่ำกว่า training loss บอกได้เลยหรือว่าไม่มี Overfitting

**เฉลย** ตรวจว่าผลชุดฝึกมาจาก train mode ที่เปิด Dropout หรือไม่ ลองวัด train ใน eval mode แล้วเปรียบเทียบแนวโน้มทั้งเส้นและผลบนข้อมูลที่แยก

## class04

[กลับไปบทเรียน](../lectures/04-cnn-basics.md)

### คำนวณ — Conv2d ใช้ 3 input channels, 16 output channels, kernel 3×3, groups 1 และ bias มีพารามิเตอร์กี่ตัว

**เฉลย** หนึ่ง filter มี 3×3×3 Weight และ Bias หนึ่งค่า รวม 28 ค่า คูณ 16 filters ได้ 448 ค่า

### คำนวณ — ภาพ 28×28 ผ่าน Conv ขนาด 3 padding 1 stride 1 แล้ว MaxPool ขนาด 2 stride 2 ได้ shape spatial เท่าใด

**เฉลย** Conv รักษาขนาด 28 เพราะ padding ชดเชยขอบที่ Kernel ใช้ Pool วางหน้าต่างไม่ทับกันขนาด 2 จึงเหลือ 14×14

## class05

[กลับไปบทเรียน](../lectures/05-cnn-classifier.md)

### วิเคราะห์ — ทำไม VGG16 ที่มี Adaptive Pooling ยังต้องจับคู่ normalization กับ weights

**เฉลย** Adaptive Pooling ช่วยแก้ shape ก่อน Linear ส่วน normalization เปลี่ยนสเกล input ตามที่ Backbone ได้รับตอนฝึก ทั้งสองแก้คนละจุด

### โค้ด — ทำไมใช้ model.eval() แต่ยังต้องเปิด gradient สำหรับ Grad-CAM

**เฉลย** eval คุมการทำงานของ layers แต่ Grad-CAM ต้องคำนวณอนุพันธ์ของ class logit ต่อ Feature Map ถ้า Forward อยู่ใน inference_mode กราฟสำหรับ backward จะไม่มี

## class06

[กลับไปบทเรียน](../lectures/06-rnn.md)

### คำนวณ — ถ้า RNN รับ (16,50,1) และ hidden size 64 แบบ batch_first shape ก่อนและหลังเลือก timestep สุดท้ายคืออะไร

**เฉลย** output ทุกเวลาเป็น (16,50,64) เลือกแกนเวลา index −1 เหลือ (16,64) แล้วหัวหนึ่ง output ให้ (16,1)

### วิเคราะห์ — ทำไม one-step error ที่ต่ำไม่รับประกัน recursive forecast ที่ดี

**เฉลย** One-step ป้อน Window จริง ส่วน recursive ป้อน prediction กลับเข้า Window ความผิดพลาดจึงกลายเป็น input ของ step ถัดไปและสะสมได้

## class07

[กลับไปบทเรียน](../lectures/07-lstm-gru.md)

### คำนวณ — LSTM มิติเดียวมี state เดิม 4, forget 0.25, input 0.5, candidate 0.6 ได้ Cell State เท่าใด

**เฉลย** ส่วนเดิมเหลือ 4×0.25=1 ส่วนใหม่คือ 0.5×0.6=0.3 รวมเป็น 1.3 การอ่าน Hidden State ยังขึ้นกับ Output Gate

### วิเคราะห์ — เหตุใด Bidirectional RNN ใช้จำแนก review ได้ แต่ต้องระวังกับ Forecast ที่มาถึงทีละจุด

**เฉลย** Review พร้อมทั้งลำดับก่อนอ่าน จึงใช้บริบทขวาและซ้ายได้ Forecast ต้องรู้ว่าตอนทำนายมีข้อมูลใดอยู่แล้ว ห้ามนำจุดอนาคตที่ยังไม่มาคำนวณ Feature

## class08

[กลับไปบทเรียน](../lectures/08-attention-transformer.md)

### คำนวณ — Cross-Attention มี Q shape (3,8), K shape (5,8) และ V shape (5,4) ตาราง score กับ output มีขนาดเท่าใด

**เฉลย** QKᵀ เป็น (3,5) ให้ score ห้า keys ต่อ query หลัง Softmax คูณ V ได้ (3,4) หนึ่งแถวต่อ query

### โค้ด — เหตุใด Transformer Lab ใช้ mask ทั้งใน Attention และตอนเฉลี่ย output

**เฉลย** Key mask กัน PAD จากการถูกอ่าน แต่ผลยังมีตำแหน่งที่เป็น padded queries ตอนเฉลี่ยจึงล้างค่าและหารด้วยจำนวน token จริง

## class09

[กลับไปบทเรียน](../lectures/09-adversarial-attack-generative-ai.md)

### พื้นฐาน — Masked Image Modeling ได้ target จากที่ไหน

**เฉลย** กันบาง patches ออกแล้วใช้ส่วนที่ถูกกันเป็นเป้าหมาย การเตรียม target มาจากภาพเองจึงทำ pretraining บนภาพที่ไม่ติดป้ายได้

### วิเคราะห์ — Accuracy บนชุดทดสอบปกติเท่ากันสองโมเดล เหตุใดจึงควรวัด Robustness เพิ่ม

**เฉลย** Accuracy สะท้อนข้อมูลและเงื่อนไขที่ใช้ประเมินเท่านั้น เมื่อต้องรับ input ที่ถูกรบกวนให้วัดตามเงื่อนไขนั้นและรายงานคู่กับผลปกติ

### คำนวณ — Triplet Loss เป็นเท่าใดเมื่อระยะ A–P = 0.4, A–N = 0.7 และ margin = 0.5

**เฉลย** วงเล็บได้ 0.4−0.7+0.5=0.2 ค่าเป็นบวกจึงได้ Loss 0.2

## class10

[กลับไปบทเรียน](../lectures/10-autoencoder.md)

### คำนวณ — ถ้า logvar เป็น 0 และ mean เป็น 0 ใน latent หนึ่งมิติ เทอม KL เป็นเท่าใด

**เฉลย** variance คือ exp(0)=1 แทนสูตรได้ −0.5×(1+0−1−0)=0 posterior มิตินี้ตรงกับ prior มาตรฐาน

### วิเคราะห์ — ทำไม AE ที่ reconstruct ดีอาจให้ภาพแปลกเมื่อ sample จาก Gaussian

**เฉลย** Reconstruction ฝึกเฉพาะ latent codes ที่ Encoder ส่งมา ตำแหน่งสุ่มจาก Gaussian อาจอยู่ในพื้นที่ที่ Decoder แทบไม่เคยเห็น ดู scatter และ grid ควบคู่กับ reconstruction

## class11

[กลับไปบทเรียน](../lectures/11-gan.md)

### พื้นฐาน — ระหว่างอัปเดต G ทำไมต้องมี Gradient ผ่าน D แม้ไม่อัปเดต D

**เฉลย** คะแนนของ D เป็นสัญญาณที่เชื่อมผลภาพจาก G กับ Loss Gradient ต้องผ่านการคำนวณของ D ไปยังผล G แต่ optimizer อัปเดตเฉพาะพารามิเตอร์ G

### วิเคราะห์ — ตัวอย่าง GAN ดูสมจริงสิบภาพแต่มีลักษณะซ้ำกันมาก ควรสงสัยอะไรและตรวจอย่างไร

**เฉลย** สงสัย Mode Collapse ลองตรวจหลาย noise draws แล้วดู diversity ของภาพและเทียบสถิติ feature กับข้อมูลจริง ภาพที่คมบางภาพแสดงคุณภาพรายตัวอย่างเพียงส่วนเดียว
