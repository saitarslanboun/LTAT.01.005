import os

classes = os.listdir("flowers")

itarget = open("data.img", "w")
ltarget = open("data.lbl", "w")
for a in range(len(classes)):
	print str(a+1) + " : " + str(len(classes))
	images = os.listdir(os.path.join("flowers", classes[a]))
	for b in range(len(images)):
		line = os.path.join("flowers", classes[a], images[b])
		if line.split(".")[-1] == "jpg":
			line += "\n"
		else:
			continue
		itarget.write(line.encode("utf-8"))
		line = classes[a] + "\n"
		ltarget.write(line.encode("utf-8"))
itarget.close()
ltarget.close()
