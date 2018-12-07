class Car():
    def __init__(self):
        self.haha = {}
    def ll(self):
        hh = self.haha
        for i in range(3):
            hh[i] = i
    def kk(self):
        print(self.haha)


if __name__ == '__main__':
    c = Car()
    c.kk()