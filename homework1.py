import jieba
import jieba.posseg as contents    #词性标注
import jieba.posseg as pseg
from collections import Counter


messages = jieba.cut("万里长城是中国古代劳动人民血汗的结晶和中国古代文化的象征和中华民族的骄傲",cut_all=False)   #精确模式
print ( '【精确模式下的分词:】'+"/ ".join(messages))

messages = jieba.cut("万里长城是中国古代劳动人民血汗的结晶和中国古代文化的象征和中华民族的骄傲",cut_all=True)   #全模式
print ( '【全模式下的分词:】'+"/ ".join(messages))

messages = jieba.cut_for_search("万里长城是中国古代劳动人民血汗的结晶和中国古代文化的象征和中华民族的骄傲")   #搜索引擎模式
print ( '【搜索引擎模式下的分词:】'+"/ ".join(messages))

messages = contents.cut('万里长城是中国古代劳动人民血汗的结晶')
for message in messages:    #使用for循环逐一获取划分后的词语
    print(message.word,message.flag)


# 读取文本文件，按行读取
with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 定义分词函数，参数为一行文本
def seg_line(line):
    words = pseg.cut(line.strip())  # strip() 去除两端空格和换行符
    return [(w.word, w.flag) for w in words]  # 保留每个词的词性

# 对每一行进行分词，结果保存在一个列表中
seg_list = []
for line in lines:
    seg_list.append(seg_line(line))

# 合并所有行的分词结果
words = []
for line_seg in seg_list:
    words += line_seg

# 统计词频
word_count = Counter(words)

# 将统计结果写入文件,格式为，词，词性，词频
with open('output.txt', 'w', encoding='utf-8') as f:
    for (word, flag), count in word_count.items():
        f.write('{},{},{}\n'.format(word, flag, count))
