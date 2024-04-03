from tkinter import *
from PIL import Image, ImageTk

root = Tk()

root.geometry("754x584") #widthxheight
root.minsize(400,300)
#root.maxsize()

root.title("MY FIRST GUI")

# Important Label Options
# text - adds the text
# bd - background
# fg - foreground
# font - sets the font
# 1. font=("comicsansms", 19, "bold")
# 2. font="comicsansms 19 bold"

# padx - x padding
# pady - y padding
# relief - border styling - SUNKEN, RAISED, GROOVE, RIDGE

Label(text="Hello World!. This is a equation solver").pack()

# img = PhotoImage(file="chaos.png")
# Label(image=img).pack()

# For jpg, jpeg etc....

img = ImageTk.PhotoImage(Image.open("pexels-jeshootscom-714698.jpg"))

Label(image=img).pack()

#In this modified code, img is a reference to the image object. As long as this reference is kept (which it is, because img is a global variable), the image will be displayed properly. This is necessary because of the way Python’s garbage collector works - it will automatically delete any object that doesn’t have a reference1234. So, by keeping a reference to the image, we ensure that it isn’t prematurely deleted.due to Python’s garbage collection. When you add a PhotoImage or other Image object to a Tkinter widget, you must keep your own reference to the image object. If you don’t, the image won’t always show up. 




root.mainloop()
