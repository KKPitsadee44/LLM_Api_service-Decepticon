# LLM_Api_service
Healthcare solution Hackathon 2025

Server Api service
# start service
 - nohup python mainAPI.py & 

#stop service
 1. Kill the existing process
  - bashkill -9 76927
 2. Clear port
  - sudo kill $(sudo lsof -t -i:5000)


How mainAPI work
- Flask API ตัวนึงที่รับคำถาม (question) จาก body ของ request แล้วเอาไปส่งต่อให้ AI model ของเรา  localhost:8000/v1 เรียกผ่านผ่าน aiohttp ทำให้ API เป็นแบบ async  แล้วพอได้คำตอบกลับมา ก็ส่ง response กลับให้ฝั่งที่เรียก API
