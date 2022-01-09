class Point:
    def __init__(self):
        self._X = 0
        self._Y = 0

    @property
    def get_X(self):
        return self._X

    @property
    def get_Y(self):
        return self._Y

    @get_X.setter
    def set_X(self, a):
        self._X = a

    @get_Y.setter
    def set_Y(self, a):
        self._Y = a

    def __str__(self):
        return '(' + str(self._X) + ',' + str(self._Y) + ')'

    def distance(self, other):
        d = Point()
        d.set_X = other.get_X - self._X
        d.set_Y = other.get_Y - self._Y
        return d


class Line(Point):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def distance(self):
        return self.x.distance(self.y)

    def on_line(self, other):
        delta_x = l.distance().get_X
        delta_y = l.distance().get_Y
        m = delta_y / delta_x
        n = self.x.get_Y - self.x.get_X * m
        if other.get_X * m + n == other.get_Y:
            return print('The point {} is on the line'.format(other))
        else:
            return print('The point {} is not on the line'.format(other))


t = Point()
t.set_Y = 1
t.set_X = 1
g = Point()
g.set_Y = 4
g.set_X = 3
print(t.distance(g))
l = Line(t, g)
f = Point()
f.set_X = 5
f.set_Y = 5
print(l.on_line(t))
print(l.on_line(f))
