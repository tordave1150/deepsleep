# ศัพท์ Deep Learning

[Index](../INDEX.md) · [Formula Reference](formula-reference.md)

| English term | ความหมายและบริบท |
| --- | --- |
| Tensor | ข้อมูลเชิงตัวเลขที่มีแกนได้หลายแกน อ่าน shape โดยระบุว่าแต่ละแกนเก็บ Batch, Feature, Channel หรือ Time ตามงาน [Class 01](../lectures/01-introduction-pytorch.md) |
| Forward Pass | การคำนวณจาก Input ผ่านชั้นต่าง ๆ จนได้ Output ค่าระหว่างทางบางส่วนถูกใช้ใน Backpropagation [Class 02](../lectures/02-mlp-basics.md) |
| Activation Function | ฟังก์ชันที่เปลี่ยนค่าหลัง Linear เช่น ReLU ช่วยให้โครงข่ายประกอบการแปลงที่ไม่เป็น Linear [Class 02](../lectures/02-mlp-basics.md) |
| Logit | คะแนนดิบจากหัวจำแนก ก่อน Sigmoid หรือ Softmax Loss บางตัวรับ logits และรวมการแปลงไว้ในการคำนวณ [Class 02](../lectures/02-mlp-basics.md) |
| Loss Function | ฟังก์ชันให้คะแนนความคลาดเคลื่อนหรือเป้าหมายระหว่างฝึก Loss กับ Metric อาจเป็นคนละตัว เช่น MSE กับ MAE ใน Lab Regression [Class 02](../lectures/02-mlp-basics.md), [Class 03](../lectures/03-mlp-problems-visualization.md) |
| Gradient | อนุพันธ์ที่บอกว่า Loss เปลี่ยนอย่างไรใกล้ค่าพารามิเตอร์ปัจจุบัน Optimizer ใช้ Gradient เพื่อคำนวณการอัปเดต [Class 02](../lectures/02-mlp-basics.md) |
| Backpropagation | ใช้ Chain Rule ย้อนกราฟเพื่อหาผลของพารามิเตอร์ต่อ Loss เมื่อมี recurrence การคำนวณย้อนผ่าน timestep ที่เชื่อมกันเรียก BPTT [Class 02](../lectures/02-mlp-basics.md), [Class 06](../lectures/06-rnn.md) |
| Batch และ Epoch | Batch คือกลุ่มตัวอย่างที่ประมวลผลพร้อมกัน Epoch คือการผ่านชุดฝึกหนึ่งรอบซึ่งอาจมีหลาย Batch [Class 02](../lectures/02-mlp-basics.md) |
| Learning Rate | ค่าที่กำหนดสเกลของก้าวปรับพารามิเตอร์ Scheduler ปรับค่าระหว่างฝึกตามเวลาหรือ Metric [Class 02](../lectures/02-mlp-basics.md) |
| Overfitting | โมเดลเข้ากับข้อมูลฝึกมากจนการประเมินข้อมูลที่กันไว้ด้อยลง การอ่านเส้น train กับ validation ร่วมกันช่วยวินิจฉัย [Class 03](../lectures/03-mlp-problems-visualization.md) |
| Regularization | การปรับวิธีฝึกเพื่อช่วย Generalization ตัวอย่างในคอร์สคือ penalty, Dropout, Early Stopping และ Data Augmentation [Class 03](../lectures/03-mlp-problems-visualization.md), [Class 05](../lectures/05-cnn-classifier.md) |
| Batch Normalization | ใช้สถิติราย feature หรือ channel และพารามิเตอร์ scale/shift การฝึกและ Inference ใช้แหล่งสถิติต่างกัน [Class 03](../lectures/03-mlp-problems-visualization.md), [Class 04](../lectures/04-cnn-basics.md) |
| Calibration | ความสอดคล้องระหว่าง confidence ที่รายงานกับอัตราถูกจริงในกลุ่มตัวอย่าง วัดต่างจาก Accuracy ของทั้งชุด [Class 03](../lectures/03-mlp-problems-visualization.md) |
| Convolution | การใช้ Kernel ร่วมกันตามตำแหน่งเพื่อคำนวณ Feature Map พิจารณา Kernel size, stride, dilation และ padding เพื่อหาขนาด output [Class 04](../lectures/04-cnn-basics.md) |
| Pooling | ย่อ spatial หรือ time axis ด้วยการรวมค่าตามหน้าต่าง เช่น ค่าสูงสุดหรือเฉลี่ย รายละเอียดตำแหน่งบางส่วนจึงหายไป [Class 04](../lectures/04-cnn-basics.md) |
| Backbone | ส่วนของโมเดลที่แปลงข้อมูลเป็น Features และนำไปต่อ Head ได้ ตอนย้ายงานต้องจับคู่ preprocessing กับ Weight ที่ใช้ [Class 05](../lectures/05-cnn-classifier.md), [Class 08](../lectures/08-attention-transformer.md) |
| Fine-tuning | ปรับพารามิเตอร์บางส่วนหรือทั้งหมดของโมเดลที่ฝึกมาก่อนเพื่อให้เข้ากับงานเป้าหมาย จำนวนชั้นที่เปิดมีผลต่อต้นทุนและขอบเขตของการปรับ [Class 05](../lectures/05-cnn-classifier.md), [Class 08](../lectures/08-attention-transformer.md) |
| Grad-CAM | ใช้ Gradient ของคะแนนคลาสถ่วง Feature Maps เป็น heatmap อ่านร่วมกับ target layer, class และภาพ input [Class 05](../lectures/05-cnn-classifier.md) |
| Hidden State | เวกเตอร์ที่ส่งบริบทจาก timestep ก่อนหน้าไปยัง step ถัดไป Hidden State มีมิติแยกจากความยาวของลำดับ [Class 06](../lectures/06-rnn.md) |
| Teacher Forcing | ใช้ target จริงของ step ก่อนเป็น Decoder Input ระหว่างฝึก ตอนทำนายแบบ autoregressive ต้องอาศัยผลที่เพิ่งทำนาย [Class 06](../lectures/06-rnn.md) |
| Gate | เวกเตอร์คูณรายมิติเพื่อคุมการไหลของข้อมูล ใน LSTM และ GRU ใช้เลือกส่วนของ state เดิมและ candidate ใหม่ [Class 07](../lectures/07-lstm-gru.md) |
| Padding Mask | ระบุตำแหน่งเสริมที่ต้องกันออกจากการอ่านหรือการรวม Representation ให้ตรวจ convention ของ mask ใน API ที่ใช้ [Class 08](../lectures/08-attention-transformer.md) |
| Self-Attention | Query, Key และ Value มาจากลำดับเดียวกัน แต่ผ่าน projection ที่เรียนรู้ได้ การผสม Value ขึ้นกับคะแนนของแต่ละ Query [Class 08](../lectures/08-attention-transformer.md) |
| Cross-Attention | Query กับ Key/Value มาจากคนละ stream เช่น Decoder อ่าน Encoder Output จำนวน Query กับ Key จึงต่างกันได้ [Class 08](../lectures/08-attention-transformer.md) |
| Self-Supervised Learning | สร้างเป้าหมายการฝึกจากข้อมูลเอง เช่น ทำนายส่วนที่ซ่อนหรือเปรียบเทียบหลายมุมมอง ตัวแทนที่ได้สามารถนำไปปรับกับงานติดป้าย [Class 09](../lectures/09-adversarial-attack-generative-ai.md) |
| Latent Space | ปริภูมิของตัวแทนที่เรียนรู้จากข้อมูล คุณสมบัติของบริเวณระหว่างจุดมีผลต่อการสุ่มและ interpolation [Class 09](../lectures/09-adversarial-attack-generative-ai.md), [Class 10](../lectures/10-autoencoder.md) |
| Reconstruction Loss | วัดความต่างระหว่าง input ที่ใช้เป็น target กับผลจาก Decoder ตัวอย่าง AE ใช้ BCE ต่อ Pixel [Class 10](../lectures/10-autoencoder.md) |
| KL Divergence | วัดความแตกต่างเชิงการแจกแจง ใน VAE ใช้ปรับ posterior ให้สอดคล้องกับ prior เพื่อช่วยการสุ่ม [Class 10](../lectures/10-autoencoder.md) |
| Reparameterization | เขียน latent sample เป็นฟังก์ชันของ mean, standard deviation และ noise ที่สุ่มแยก จึงคงเส้นทางอนุพันธ์ไปยังพารามิเตอร์ Encoder [Class 10](../lectures/10-autoencoder.md) |
| Mode Collapse | Generator ส่ง noise หลายจุดไปยัง output ที่คล้ายกันมาก ความสมจริงบางภาพจึงอยู่ร่วมกับ diversity ต่ำได้ [Class 11](../lectures/11-gan.md) |
| Fréchet Inception Distance | เปรียบเทียบสถิติ feature distributions ของภาพจริงและภาพสร้าง ตัวสกัด, preprocessing และจำนวนตัวอย่างมีผลต่อความเปรียบเทียบได้ [Class 11](../lectures/11-gan.md) |
