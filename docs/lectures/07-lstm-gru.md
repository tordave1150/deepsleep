# 7. LSTM, GRU และทางเลือกสำหรับข้อมูลลำดับ

[← ก่อนหน้า](06-rnn.md) | [Index](../INDEX.md) | [ถัดไป →](08-attention-transformer.md)

## แหล่งเรียน

class07 · ตรวจต้นฉบับ 2026-11-14

- [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view)
- Lab [LSTM_ex1](../labs/LSTM_ex1.md)
- Lab [LSTM_ex3](../labs/LSTM_ex3.md)

## เป้าหมายและพื้นฐาน

อ่าน LSTM และ GRU ทีละ Gate และเลือกวิธีจัด state ให้เข้ากับโจทย์ ถ้ายังสับสนว่า timestep ต่างจาก layer อย่างไร ให้ย้อนดู [Class 06](06-rnn.md)

## ปัญหาลำดับยาวและวิธีคุม Gradient

เมื่อ unroll ยาวขึ้น BPTT ต้องย้อนผ่านการคูณอนุพันธ์หลายครั้ง กรณีในสไลด์ใช้คำที่อยู่ห่างกันเพื่อแสดงว่าความสัมพันธ์ต้นประโยคอาจเหลือสัญญาณน้อยเมื่อถึงท้าย Gradient Clipping จำกัดขนาด Gradient ก่อน optimizer.step() เพื่อรับมือการระเบิด แบบ clip ตามค่าเปลี่ยนบางองค์ประกอบ ส่วนแบบตาม norm ย่อทั้งเวกเตอร์เพื่อรักษาทิศโดยรวม

Layer Normalization หาสถิติข้าม features ของตัวอย่างที่กำลังคำนวณ ภาพ BN/LN แสดงการจัดกลุ่มสถิติคนละแกน Truncated BPTT ส่งค่าของ state ข้าม chunk ได้แต่ตัดกราฟอนุพันธ์ที่ขอบ ลดหน่วยความจำแลกกับ Gradient ที่ไม่ย้อนข้ามขอบนั้น หาก Loss มีเฉพาะปลายลำดับ ช่วงต้นอาจไม่ได้รับสัญญาณจาก Loss ปลายทาง

> **อ้างอิง** [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 2–9

<a name="lstm-gates"></a>

## LSTM และ Cell State

รูป LSTM แสดงเส้นแนวนอนของ Cell State $C_t$ และเส้น Hidden State $h_t$ ที่ส่งผลออกไป แทนการคำนวณ Gate ด้วย $s_f=W_{f,x}x_t+W_{f,h}h_{t-1}+b_f$ และนิยาม $s_i,s_o,s_g$ แบบเดียวกันโดยใช้ Weight และ Bias ของ Gate นั้น จะได้ $f_t=\sigma(s_f),i_t=\sigma(s_i),o_t=\sigma(s_o),g_t=\tanh(s_g)$

$$
C_t=f_t\odot C_{t-1}+i_t\odot g_t,\qquad
h_t=o_t\odot\tanh(C_t).
$$

สัญลักษณ์ $\odot$ คือคูณตำแหน่งเดียวกัน ทุก Gate มีมิติเท่ากับ state สำหรับตัวอย่างหนึ่ง Forget Gate คุมส่วนเดิม Input Gate คุม candidate และ Output Gate คุมสิ่งที่เผยจาก state ให้ชั้นถัดไป ในภาพการอ่านประโยค การเปลี่ยนหัวเรื่องทำให้มีเหตุผลที่จะลดน้ำหนักข้อมูลเดิมและเพิ่ม candidate ใหม่

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** มิติหนึ่งมี $C_{t-1}=0.8,f_t=0.5,i_t=0.25,g_t=0.4$ จึงได้ $C_t=0.5$ หาก $o_t=0.2$ ค่าที่ส่งออกคือ $0.2\tanh(0.5)$ ส่วนที่ส่งบนเส้น Cell State ยังคงเป็น $0.5$

> **อ้างอิง** [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 10–16

## GRU และการผสม Hidden State

GRU ในภาพรวม state เป็นเวกเตอร์เดียว Reset Gate $r_t$ คุมข้อมูลเดิมที่เข้าสู่ candidate $g_t$ ส่วน Update Gate $z_t$ กำหนดสัดส่วน old/new ตามสัญกรณ์ของ handout

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot g_t.
$$

เมื่อ $z_t$ ใกล้ 1 จะรักษาข้อมูลเดิม เมื่อใกล้ 0 จะให้น้ำหนัก candidate มากขึ้น ภาพเปรียบเทียบชี้ว่า GRU มีการคำนวณ Gate น้อยกว่า LSTM และชวนทดสอบทั้งสองกับงานจริง หากอ่านสูตรจาก library ให้ตรวจสัญกรณ์และตำแหน่งของ reset ใน implementation ก่อนเทียบกับภาพ

> **อ้างอิง** [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 17–21

## Direction และการจัดการ State

Uni-directional RNN ใช้บริบททางอดีต เหมาะเมื่อข้อมูลอนาคตยังไม่มา Bidirectional อ่านไปและย้อนกลับแล้วนำ Features สองทางมารวม เหมาะเมื่อมีลำดับครบก่อนตัดสิน เช่น อ่าน review ทั้งข้อความ แต่หากทำนายแบบ online ต้องตรวจว่าข้อมูลอนาคตพร้อมจริงหรือไม่

Stateless ใช้ state ใหม่ระหว่างลำดับอิสระ ส่วน Stateful ส่ง state ต่อระหว่าง chunks ของลำดับเดิม ใน PyTorch ต้องส่ง state ให้ถูก stream และจัดการ detach เมื่อใช้การตัดกราฟ หากสลับตำแหน่งตัวอย่างข้าม Batch โดยไม่จัด state ให้ตรงกัน บริบทอาจไปปะปนระหว่างตัวอย่าง

> **อ้างอิง** [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 22–26

## Labs และทางเลือกนอก Recurrent Cells

Lab ทำนาย sine และ composite sine เปรียบเทียบผล one-step กับ recursive และเก็บภาพตาม Epoch Lab IMDB ใช้ BiLSTM, Embedding และ packing เพื่ออ่าน review ความยาวต่างกัน สไลด์ยังยกการอ่าน MNIST ทีละแถวเป็น 28 timesteps และ ConvLSTM สำหรับวิดีโอ Conv1d ตามลำดับใช้ Kernel ตรวจรูปแบบตามเวลา ส่วน WaveNet ใช้ Causal Convolution และ dilation เพิ่มระยะมองโดยไม่ใช้อนาคต ภาพท้ายบทแนะนำ Foundation Models สำหรับ Time Series เช่น TimeGPT และ TSMamba

> **อ้างอิง** [class07/7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 26–35

## ทบทวนและเชื่อมบท

LSTM ใช้เส้น Cell State แยกและ GRU ผสมค่าเดิมกับ candidate ใน Hidden State เดียว เลือกวิธีวัดให้ตรงกับบริบทที่มีจริงและ horizon ที่ต้องการ ต่อไป [Class 08](08-attention-transformer.md) ให้ Decoder อ่าน encoder states หลายตำแหน่งโดยตรง

## คำถามทบทวน

1. คำนวณ Cell State เมื่อค่าเดิมเป็น 4, Forget Gate เป็น 0.25, Input Gate เป็น 0.5 และ candidate เป็น 0.6
2. เมื่อไรที่ Bidirectional RNN ใช้บริบทฝั่งขวาได้

[คำถามและเฉลย](../review/practice-and-solutions.md#class07) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| ปัญหาลำดับยาวและวิธีคุม Gradient | [7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 2–9 |
| LSTM และ Cell State | [7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 10–16 |
| GRU และการผสม Hidden State | [7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 17–21 |
| Direction และการจัดการ State | [7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 22–26 |
| Labs และทางเลือกนอก Recurrent Cells | [7_LSTM_GRU_else_2handout_color.pdf](https://drive.google.com/file/d/1efJ8_jhVgJcrRgoVaqnJx20GU523w1T_/view) · PDF หน้า 26–35 |

[← ก่อนหน้า](06-rnn.md) | [Index](../INDEX.md) | [ถัดไป →](08-attention-transformer.md)
