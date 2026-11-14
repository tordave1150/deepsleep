# สมการสำหรับทบทวน

[Index](../INDEX.md) · [Practice & Solutions](practice-and-solutions.md)

## Linear, Softmax และ Cross-Entropy

$$
z=Wx+b,\qquad p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad L_\mathrm{CE}=-\sum_i y_i\log p_i.
$$

$W$ มีหนึ่งแถวต่อ output unit และหนึ่งคอลัมน์ต่อ input feature Softmax แปลง logits เป็น distribution ต่อหนึ่งตัวอย่าง ใน CE ถ้า $y$ เป็น one-hot จะเหลือเทอมคลาสจริงหนึ่งเทอม PyTorch ใช้ logits กับ CrossEntropyLoss ใน Lab

Binary CE ใช้ $-y\log p-(1-y)\log(1-p)$ เมื่อ $p$ คือความน่าจะเป็นของคลาสบวก [Class 02](../lectures/02-mlp-basics.md)

## Gradient Descent และค่าเฉลี่ยตาม Batch

$$
w\leftarrow w-\eta\nabla_w L,\qquad \bar L=\frac{\sum_b n_bL_b}{\sum_b n_b}.
$$

$\eta$ กำหนดขนาดก้าว ส่วน $L_b$ คือค่าเฉลี่ยใน Batch ที่มี $n_b$ ตัวอย่าง เทอม $n_b$ ทำให้แต่ละตัวอย่างได้สัดส่วนเท่ากันแม้ Batch สุดท้ายเล็กกว่า [Class 02](../lectures/02-mlp-basics.md) · [MLP1](../labs/MLP1.md)

Adam ใช้ $\hat m_t$ จากค่าเฉลี่ย Gradient และ $\hat v_t$ จากค่าเฉลี่ย Gradient กำลังสอง หลัง Bias Correction สเกลก้าวรายพารามิเตอร์คำนวณจาก $\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ โดยทิศขึ้นกับเครื่องหมายของ $\hat m_t$ [Class 02](../lectures/02-mlp-basics.md)

## Batch Normalization และ Regularization

$$
\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y_i=\gamma\hat x_i+\beta .
$$

$\mu_B,\sigma_B^2$ คือสถิติของ Batch, $\epsilon$ ช่วยเรื่องตัวหาร, $\gamma$ กับ $\beta$ เรียนรู้ได้ Eval ใช้ running statistics ที่สะสมไว้ L1 เพิ่ม $\lambda\sum_j|w_j|$ และ L2 เพิ่ม $\lambda\sum_j w_j^2$ ใน Loss [Class 03](../lectures/03-mlp-problems-visualization.md)

## Conv2d และ Grad-CAM

$$
H_\text{out}=\left\lfloor\frac{H+2P-D(K-1)-1}{S}+1\right\rfloor.
$$

$H,K,P,D,S$ แทน input length, kernel length, padding ข้างละหนึ่งค่า, dilation และ stride ตามลำดับ ใช้แยกกับแกนกว้าง จำนวน Weight ของ Conv2d แบบ groups $g$ คือ $C_\mathrm{out}(C_\mathrm{in}/g)K_hK_w$ ถ้าเปิด bias ให้เพิ่ม $C_\mathrm{out}$ [Class 04](../lectures/04-cnn-basics.md)

$$
\alpha_k^c=\frac1{HW}\sum_{i,j}\frac{\partial s_c}{\partial A^k_{ij}},
\qquad H_c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right).
$$

$s_c$ คือ class logit ส่วน $A^k$ คือ Feature Map ช่อง $k$ Weight หนึ่งค่าต่อช่องได้จาก spatial average ของ gradient [Class 05](../lectures/05-cnn-classifier.md)

## Recurrent State และ Gates

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b).
$$

$x_t$ เป็น input ของเวลา $t$ และ $h_t$ มีขนาด hidden size BPTT รวม Gradient จากทุกจุดที่ Weight ชุดเดียวกันถูกใช้ [Class 06](../lectures/06-rnn.md)

$$
C_t=f_t\odot C_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(C_t).
$$

LSTM ใช้ forget, input และ output gates ต่อมิติของ state ส่วน $g_t$ เป็น candidate ใน GRU ใช้ $h_t=z_t\odot h_{t-1}+(1-z_t)\odot g_t$ ตาม convention ของสไลด์ [Class 07](../lectures/07-lstm-gru.md)

## Attention และ LoRA

$$
A=\operatorname{softmax}_{\text{keys}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad
W=W_0+BA .
$$

สมการแรกใช้ $A$ เป็น attention output โดย $M$ คือ additive mask และ $d_k$ คือมิติของ Key สมการ LoRA ใช้ $A,B$ เป็น low-rank factors ของ Weight update ซึ่งเป็นคนละสัญลักษณ์กับ attention output [Class 08](../lectures/08-attention-transformer.md)

## Triplet Loss

$$
L=\max(0,d(A,P)-d(A,N)+m).
$$

$A,P,N$ คือ Anchor, Positive และ Negative ตามลำดับ หาก Negative ไกลกว่า Positive อย่างน้อย $m$ แล้ว Loss เป็นศูนย์ [Class 09](../lectures/09-adversarial-attack-generative-ai.md)

## VAE

$$
z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I),\qquad
L=L_\mathrm{recon}+\beta L_\mathrm{KL}.
$$

$$
L_\mathrm{KL}=-\frac12\sum_j(1+\log\sigma_j^2-\sigma_j^2-\mu_j^2).
$$

สูตร KL ใช้ posterior Gaussian แบบ diagonal เทียบกับ prior มาตรฐาน Lab ใช้ BCE ต่อภาพเป็น reconstruction และแยก weighted KL ออกมาอ่าน [Class 10](../lectures/10-autoencoder.md)

## GAN

$$
\min_G\max_D\bigl(\mathbb E_x[\log D(x)]+\mathbb E_z[\log(1-D(G(z)))]\bigr).
$$

$x$ มาจากข้อมูลจริงและ $z$ มาจาก prior ของ noise แบบ non-saturating ฝั่ง G ลด $-\mathbb E_z[\log D(G(z))]$ เพื่อให้สัญญาณฝึกช่วงต้นใช้งานได้ [Class 11](../lectures/11-gan.md)

Inception Score คือ $\exp(\mathbb E_x[D_\mathrm{KL}(p(y\mid x)\Vert p(y))])$ โดย $x$ เป็นภาพที่สร้างและ $p(y)$ เป็น distribution ของคลาสที่ classifier ให้โดยเฉลี่ยในชุดภาพสร้าง [Class 11](../lectures/11-gan.md)
