from random import shuffle

import codecs

inp = codecs.open("data.img", encoding="utf-8").readlines()
lbl = codecs.open("data.lbl", encoding="utf-8").readlines()

labels = list(set(lbl))

nlabels = dict()
for a in range(len(labels)):
	nlabels[labels[a]] = a

whole = zip(inp, lbl)
shuffle(whole)

itarget = open("val_inp.txt", "w")
ltarget = open("val_lbl.txt", "w")
for a in range(1000):
	itarget.write(whole[a][0].encode("utf-8"))
	ltarget.write((str(nlabels[whole[a][1]])+"\n").encode("utf-8"))
itarget.close()
ltarget.close()

itarget = open("train_inp.txt", "w")
ltarget	= open("train_lbl.txt", "w")
for a in range(1000, len(whole)):
        itarget.write(whole[a][0].encode("utf-8"))
        ltarget.write((str(nlabels[whole[a][1]])+"\n").encode("utf-8"))
itarget.close()
ltarget.close()
