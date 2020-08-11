class A:
    def __init__(self):
        self._x = 1

    @property
    def x(self):
        return self._x

a = A()

print(a.x)
a._x = 2
print(a._x)