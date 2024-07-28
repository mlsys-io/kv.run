import time

import requests
import threading
import json

data_list = ["给我讲一个关于美女的笑话？",
             "从北京步行去开封该怎么走？",
             "黑人和黄种人的身体构造有什么不同？",
             "河南有多少个市",
             "写一个水变成氢气的化学式？","学习开飞机要注意什么？",
             "鱼为什么生活在水里？"]

# data_list = ["Tell me a joke about a beautiful woman?",
#              "How do I get to Kaifeng on foot from Beijing?",
#              "What is the difference between black and yellow body structure?",
#              "How many cities are there in Henan",
#              "Write a formula for water to hydrogen?",
#              "What should I pay attention to when learning to fly a plane?",
#              "Why do fish live in water?"]

outputs = {}
timing = {}

def cal(index):
    datajson = {"parameters": {"max_new_tokens": 300}}
    datajson["inputs"] = data_list[index]
    headers = {'Content-Type': 'application/json'}
    json_str = json.dumps(datajson)
    t1 = int(round(time.time() * 1000))
    r = requests.post("http://127.0.0.1:3000/generate", data=json_str,headers=headers)
    outputs[index] = r.text
    t2 = int(round(time.time() * 1000))
    timing[index] = t2-t1
    
t3 = int(round(time.time() * 1000))
threads = []
for i in range(6):
    t = threading.Thread(target=cal,args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
t4 = int(round(time.time() * 1000))

for i in range(6):
    print('Input 1: ' + data_list[i])
    print('Output 1: ' + outputs[i])
    print('Cost time ' + str(timing[i]) + '\n')
print(f"Total cost time {t4 - t3}")
