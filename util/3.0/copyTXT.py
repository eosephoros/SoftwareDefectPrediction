fp = open('CM1.txt', 'r') #写入内容
for line in fp:
    fq = open('noCTCM1.txt', 'a')  # 这里用追加模式 #被写入的文件
    fq.write(line)
fp.close()
fq.close()

#b写入a