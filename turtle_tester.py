import turtle
from pickle import dump
s=turtle.Screen()

n=28 # change grid size here
turtle.delay(0)

x_points=[[x for x in range(y,n*n,n)] for y in range(n)]
y_points=[[y*n+x for x in range(n)] for y in range(n)][::-1]

centers,points=[],[]

edge,diff=-300.0,600.0/n
end_len=-300.0+(diff/2)

for x in range(n):
    for y in range(n):

        centers.append((end_len+(diff*y),-(diff*x)-end_len))

    points.append((edge,edge+diff))
    edge+=diff

def square_number(x,y):
    horizontals,verticals=[],[]

    for p in range(len(points)):
        point=points[p]

        if x>point[0] and x<=point[1]: horizontals=x_points[p]
        if y>point[0] and y<=point[1]: verticals=y_points[p]

    for sq in horizontals:
        if sq in verticals:
            return sq

class Square(turtle.Turtle):
    def __init__(self,sq):
        turtle.Turtle.__init__(self)
        self.speed(10)
        self.shape('square'); self.color('white'); self.pu()
        self.goto(centers[sq][0],centers[sq][1])
        self.ondrag(drag); self.onrelease(release)

def drag(x,y):
    sq=square_number(x,y)
    Square.squares[sq]=1; turtles[sq].color('black')

def release(x,y):
    f=open('turtle.dat','wb')
    dump(Square.squares,f)
    f.close()

turtles=[Square(x) for x in range(n*n)]
Square.squares=[0]*(n*n)

s.listen()
turtle.mainloop()