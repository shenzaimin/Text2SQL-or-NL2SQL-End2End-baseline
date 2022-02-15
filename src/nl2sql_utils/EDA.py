import random

def insert_punctuation_marks(sentence):
  #PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']       #english
  PUNCTUATIONS = ['。', '，', '！', '？', '；', '：']       #chinese
  PUNC_RATIO = 0.3

  # words = sentence.split(' ')        # english
  words = list(sentence)               # chinese
  print(words)
  new_line = []
  q = random.randint(1, int(PUNC_RATIO * len(words) + 1))
  qs = random.sample(range(0, len(words)), q)

  for j, word in enumerate(words):
    if j in qs:
      new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
      new_line.append(word)
    else:
      new_line.append(word)
  #new_line = ' '.join(new_line)          # english
  new_line = ''.join(new_line)            # chinese
  return new_line

sent = '国网最早什么时候投运火力发电厂？'
new_sent = insert_punctuation_marks(sent)
print(new_sent)