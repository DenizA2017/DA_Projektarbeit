import os

for i, filename in enumerate(os.listdir('.')):
    os.rename(filename,'Fliese'+ str(i) + ".jpg")