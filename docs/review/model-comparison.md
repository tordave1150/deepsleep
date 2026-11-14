# เปรียบเทียบ Models ในคอร์ส

[Index](../INDEX.md) · [Formula Reference](formula-reference.md)

| Model | Input และจุดประสงค์ | กลไกหลัก | สิ่งที่ต้องดูเวลาเลือก | บทเรียน |
| --- | --- | --- | --- | --- |
| MLP | เวกเตอร์เพื่อ Regression หรือ Classification | Linear และ nonlinear Activation ซ้อนชั้น | ขนาด input, output และ Loss ต้องเข้ากับโจทย์ การ Flatten ภาพเสียการจัดพื้นที่ใน tensor | [Class 02](../lectures/02-mlp-basics.md), [Class 04](../lectures/04-cnn-basics.md) |
| KAN | เวกเตอร์เพื่อแทนฟังก์ชัน | ฟังก์ชันแบบ spline เรียนรู้ที่ขอบ | อ่านฟังก์ชันรายขอบและปรับ grid ได้ มีต้นทุนฝึกและข้อจำกัดตามข้อมูล | [Class 03](../lectures/03-mlp-problems-visualization.md) |
| CNN | ภาพหรือข้อมูลที่มีโครงสร้างตามตำแหน่ง | Kernel ที่ใช้ซ้ำตามตำแหน่งและ downsampling | จำนวน channels, spatial shape, memory และความละเอียดที่งานต้องรักษา | [Class 04](../lectures/04-cnn-basics.md) |
| VGG | ภาพเพื่อจำแนกหรือเป็น pretrained Backbone | ซ้อน 3×3 Conv และต่อ classifier | หัวเดิมรับ Flatten ขนาดคงที่ รุ่นใน Lab ใช้ Adaptive Pooling ก่อนหัว | [Class 05](../lectures/05-cnn-classifier.md), [CNN2_ex3](../labs/CNN2_ex3.md) |
| Inception | ภาพเพื่อจำแนก | ใช้ filter หลายขนาดแบบขนานและ 1×1 Conv ลด channels | เปรียบเทียบต้นทุนของแต่ละแขนงและรูปแบบ Feature ที่รวมได้ | [Class 05](../lectures/05-cnn-classifier.md) |
| ResNet | ภาพเพื่อเรียน representation หรือจำแนก | Residual connections บวกสัญญาณข้ามบล็อก | เปรียบเทียบตัววัดบนฮาร์ดแวร์และ input size เดียวกัน | [Class 05](../lectures/05-cnn-classifier.md) |
| DenseNet | ภาพเพื่อจำแนก | ต่อ features จากชั้นก่อนด้วย concatenation | เส้นทางสัญญาณสั้นขึ้นและใช้ features ซ้ำ | [Class 05](../lectures/05-cnn-classifier.md) |
| Xception | ภาพเพื่อจำแนก | Depthwise Separable Convolution แยก spatial filtering กับ channel mixing | ลดการคำนวณโดยแยก Weight เป็นสองขั้นที่ทำหน้าที่ต่างกัน | [Class 05](../lectures/05-cnn-classifier.md) |
| EfficientNet | ภาพเพื่อจำแนก | ปรับ depth, width และ resolution ร่วมกัน | ประสิทธิภาพขึ้นกับทรัพยากรและขนาด input | [Class 05](../lectures/05-cnn-classifier.md) |
| Vanilla RNN | ลำดับ ความยาว input/output ขึ้นกับงาน | ใช้ Weight ซ้ำและส่ง Hidden State | BPTT ยาวเกิดปัญหา gradient และ recursive forecast สะสม error ได้ | [Class 06](../lectures/06-rnn.md) |
| LSTM | ลำดับที่ต้องเก็บบริบทข้ามเวลา | Cell State แยกจาก Hidden State และสาม gates | เก็บบริบทได้นานขึ้นตามโจทย์ และใช้การคำนวณ gates เพิ่ม | [Class 07](../lectures/07-lstm-gru.md) |
| GRU | ลำดับสำหรับ Forecasting หรือ Classification | Reset/Update Gates และ Hidden State เดียว | เปรียบเทียบกับ LSTM ตามชุดข้อมูลและ protocol | [Class 07](../lectures/07-lstm-gru.md) |
| WaveNet | ลำดับที่ต้องคงความเป็น causal | ซ้อน Conv1d แบบ dilated | dilation เพิ่ม receptive field โดยไม่ต้องเพิ่มขนาด Kernel ทุกชั้น | [Class 07](../lectures/07-lstm-gru.md) |
| Transformer | ลำดับ Token สำหรับการแทนข้อมูลหรือสร้างลำดับ | Attention, position, feed-forward และ mask | Full Attention มีตารางตามคู่ Token ต้นทุนจึงโตเร็วตามความยาว | [Class 08](../lectures/08-attention-transformer.md) |
| BERT | ข้อความทั้งลำดับสำหรับ contextual representation | Bidirectional Transformer Encoder | ใช้ tokenizer ที่ตรงกับ backbone และเลือกว่าจะ freeze ชั้นใด | [Class 08](../lectures/08-attention-transformer.md) |
| ViT และ Swin | ภาพในรูปของ patch tokens | ViT ใช้ encoder ของ Transformer ส่วน Swin ใช้ shifted windows | ขนาดภาพและจำนวน patches มีผลต่อต้นทุน Attention | [Class 08](../lectures/08-attention-transformer.md) |
| Mamba | ลำดับยาว | Selective State Space Model | state แบบสรุปและ scan เปลี่ยน trade-off ระหว่างความจำกับการเข้าถึงตำแหน่ง | [Class 08](../lectures/08-attention-transformer.md) |
| AE | input เป็น target ใน Reconstruction | Encoder, bottleneck และ Decoder | reconstruction ดีได้โดย latent ยังสุ่มได้ไม่ทั่ว | [Class 10](../lectures/10-autoencoder.md) |
| VAE | input เพื่อเรียน posterior และ Reconstruction | Gaussian latent, KL และ reparameterization | สัดส่วน KL กับ reconstruction มีผลต่อรายละเอียดและโครงสร้าง latent | [Class 10](../lectures/10-autoencoder.md) |
| GAN | noise และเงื่อนไขตามรุ่น เพื่อสังเคราะห์หรือแปลงภาพ | Generator กับ Discriminator ฝึกสลับกัน | ตรวจเสถียรภาพ, Mode Collapse และความครอบคลุมตัวอย่าง | [Class 11](../lectures/11-gan.md) |
| Pix2Pix และ CycleGAN | ภาพจากสอง domain | Paired conditional training หรือ cycle consistency | เลือกตามว่ามีคู่ภาพตรงกันหรือมีเพียงภาพคนละชุด | [Class 11](../lectures/11-gan.md) |
