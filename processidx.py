import os
import fileinput
import cPickle as pickle

idxs = []
labels = []
data = []
i = 1 
for line in fileinput.input("test_videos_p.txt"):
    idxs.append(range(i,18+i))
    video,label = line.split(" ")
    labels.append(int(label))
    i+=18

print idxs
data.append(idxs)
data.append(labels)

with open("test_idx.pkl","wb") as fb:
	pickle.dump(data,fb)
