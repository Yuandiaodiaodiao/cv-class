def exe(f):

    f()


for i in range(3):
    def fx():
        print(i)
    exe(fx)
