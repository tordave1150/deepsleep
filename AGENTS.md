# แนวทางแก้ไข Deep Learning Knowledge Base

- ใช้เฉพาะแหล่งที่ตรวจอ่านได้จากโฟลเดอร์ Drive ตาม [Sources](sources/README.md) สำรวจโฟลเดอร์ย่อยใหม่เมื่อรายการเปลี่ยน อัปเดต [Manifest](sources/manifest.json), [Coverage](sources/coverage.md) และ [Status](docs/STATUS.md)
- อธิบายเป็นภาษาไทยโดยรักษา English technical terms ให้สม่ำเสมอ เริ่มจาก intuition แล้วตามรูป สมการ notation และ shapes ไปยัง implementation เชื่อมบทก่อนหน้าและถัดไป
- อ้างหน้า PDF จริงโดยเริ่มที่ 1 และนับ Notebook cell จาก 1 รวม Markdown และ Code ตรวจภาพ สูตร และกราฟคู่กับข้อความ ระบุข้อจำกัดเมื่อยืนยันจากแหล่งเรียนไม่ได้
- อธิบาย Lab จากลำดับ data, model, loss, evaluation และ output ตามต้นฉบับ ยกเฉพาะ snippet ที่จำเป็น ระบุผลเดิม ผลที่รันจริง และพฤติกรรมที่อนุมานจากโค้ดให้ชัด
- ติดป้าย Illustrative Example สำหรับตัวอย่างที่สร้างเอง หากจำเป็นต้องใช้ External Supplement ให้ระบุชื่อแหล่ง URL วันเข้าถึง และประเด็นที่เสริม
- เก็บไฟล์ดิบ ผลสกัด API keys และ outputs ไว้ในตำแหน่งที่ Git ignore ตรวจคำสั่งดาวน์โหลด ติดตั้ง เขียนไฟล์ และเรียกบริการก่อนรัน Notebook
- ตรวจ relative links, headings, fenced code, math และตำแหน่งอ้างอิงก่อนอัปเดตสถานะ ระบุ partial พร้อมขอบเขตจริงเมื่อพบช่องว่าง
