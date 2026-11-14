# เส้นทางเรียนและทบทวน

[Index](../INDEX.md) · [Practice & Solutions](practice-and-solutions.md)

## เรียนครั้งแรก

1. [Class 01](../lectures/01-introduction-pytorch.md) → [Class 02](../lectures/02-mlp-basics.md) → [Class 03](../lectures/03-mlp-problems-visualization.md) ทำ Lab setup ก่อน จากนั้นวาด loop ของ MLP และอ่านกราฟ train/validation จนระบุได้ว่าเลือก checkpoint ด้วยข้อมูลชุดไหน
2. [Class 04](../lectures/04-cnn-basics.md) → [Class 05](../lectures/05-cnn-classifier.md) คำนวณ shape ของ Conv และ Pool ก่อนเทียบ MLP กับ CNN บน MNIST จากนั้นติดตามการเปลี่ยนหัว VGG16 และอธิบายว่า Grad-CAM ต้องใช้ Gradient จากชั้นใด
3. [Class 06](../lectures/06-rnn.md) → [Class 07](../lectures/07-lstm-gru.md) → [Class 08](../lectures/08-attention-transformer.md) วาด state ข้ามเวลา แยก Loss ที่คำนวณทุก timestep จาก Loss ที่ปลาย Window แล้วเปลี่ยนมามองตาราง Query × Key และ padding mask
4. [Class 09](../lectures/09-adversarial-attack-generative-ai.md) → [Class 10](../lectures/10-autoencoder.md) → [Class 11](../lectures/11-gan.md) ระบุว่าการฝึกได้สัญญาณจาก label, input, prior หรือ Discriminator แล้วเลือกหลักฐานที่ใช้ประเมิน output แต่ละแบบ

## ทบทวนเร็ว

เปิด [Concept Map](concept-map.md) แล้วใช้ [Formula Reference](formula-reference.md) อ่าน input, output และมิติของสมการตามเส้นทางเดียวกัน ปิดท้ายด้วยคำถามของแต่ละ Class ใน [Practice & Solutions](practice-and-solutions.md) ถ้าตอบได้เฉพาะชื่อโมเดล ให้ลองเพิ่มเหตุผลหนึ่งขั้นว่าข้อมูลไหลไปที่ไหนหรือ metric มาจากชุดใด

## ทบทวนลึก

เริ่มจากสามคู่ทดลอง [MLP กับ CNN](../labs/CNN1_ex4.md), [One-step กับ Recursive Forecast](../labs/RNN_ex1_ex2_ex3.md) และ [AE กับ VAE](../labs/AE_ex1.md) ต่อคู่ให้จด input shape, target, objective, split, สิ่งที่เปลี่ยน และสิ่งที่เก็บไว้ แล้วเทียบกับ [Model Comparison](model-comparison.md) เพื่อดูว่าข้อสรุปใช้ได้ในขอบเขตใด
