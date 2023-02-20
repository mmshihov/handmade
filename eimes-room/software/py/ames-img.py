# Это программа-генератор искаженных картинок на стенах, потолке и поле комнаты Эймса.
# Она сделана на основе программы ames.py, расположенной в том же каталоге.

# Обозначения элементов комнаты Эймса 
# возьмем как на рисунке:
# "../../../eimes-room/beamer/figs/ames-xyz.png" 
# 
# Обозначения углов комнаты:
#   E = левый--нижний--ближний
#   F = левый--нижний--дальний
#   A = левый--верхний-ближний
#   B = левый--верхний-дальний
#   H = правый-нижний--ближний
#   G = правый-нижний--дальний
#   D = правый-верхний-ближний
#   C = правый-верхний-дальний
#
# Т.е.:
#   floor       - пол (EFGH);
#   ceil        - потолок (ABCD);
#   leftWall    - левая стена (EABF);
#   frontWall   - передняя стена (FBCG);
#   rightWall   - правая стена (GCDH);
# 
# Одноименные файлы выкроек вы получите на выходе.

# Трехментые координаты будут представлены списком из трех чисел.
# Двумерные (для выкроек стен, потолка и пола комнаты) --- списком из двух чисел.
# 3D точка: [x, y, z], 2D: [x, y].
# Измерения: x - ширина (право-лево), y - высота (верх-низ), z - глубина (вблизи-вдали)
# Измеряем всё в миллиметрах!

# Программа не проверяет влезет ли выкройка на A4-й лист бумаги --- подгоняйте сами.

# Точку зрения V поместим в начало координат: 

V = [0, 0, 0]

# Плоскости потолка и пола параллельны плоскости xOz; 
#  плоскости левой и правой стен параллельны yOz; 
#  передняя стена параллельна xOy.

# Чтобы задать положение комнаты в 3D, зададим координаты точки 

E = [-100, -100, 0]

# т.е. E --- это левый-нижний-ближний угол комнаты. А также измерения комнаты:

WIDTH  = 200
HEIGHT = 200
DEPTH  = 200

# Координаты остальных вершин комнаты получаются автоматически от базовой точки E:
#         + WIDTH,       + HEIFHT,       +  DEPTH
A = [E[0],          E[1] + HEIGHT,  E[2]]
D = [E[0] + WIDTH,  E[1] + HEIGHT,  E[2]]
H = [E[0] + WIDTH,  E[1],           E[2]]

F = [E[0],          E[1],           E[2] + DEPTH]
B = [E[0],          E[1] + HEIGHT,  E[2] + DEPTH]
C = [E[0] + WIDTH,  E[1] + HEIGHT,  E[2] + DEPTH]
G = [E[0] + WIDTH,  E[1],           E[2] + DEPTH]

# Также не проверяется видимость всей комнаты из точки V (проверяйте сами)

# Элемент узора (на который будет "натягиваться" картинка) в этой программе 
# представляет собой замкнутую фигуру из нескольких 2D или 3D точек (хранимых списком), 
# лежащих в одной плоскости:

class Pattern:
    # picturePath - путь к картинке
    # points - точки
    def __init__(self, picturePath: str, points: list[list[float]]):
        self.picturePath = picturePath
        self.points = points


# Определим несколько функций для генерации узоров комнаты:

# пол
def generatePatternsFloor(): 
    return []

# потолок
def generatePatternsCeil():
    return [] # todo: светильники можно сделать катринками

# левая стена
def generatePatternsLeftWall():
    return [] # todo: двери, картины можно сделать картинками

# правая стена
def generatePatternsRightWall():
    return [] # todo: двери, картины можно сделать картинками

# дальняя стена
def generatePatternsFrontWall():
    # камин
    patterns = [Pattern("input/pictures/camin-1.png", [B, C, G, F])] # TODO: пока на стену целиком
    return patterns

# сохраняем узоры в соответствующих переменных:
floorPatterns       = generatePatternsFloor()
ceilPatterns        = generatePatternsCeil()
leftWallPatterns    = generatePatternsLeftWall()
rightWallPatterns   = generatePatternsRightWall()
frontWallPatterns   = generatePatternsFrontWall()

# функция получения вектора из двух точек
def vector3D(a, b):
    return [(b[0] - a[0]),  (b[1] - a[1]),  (b[2] - a[2])]

# Определим функцию, выполняющую скалярное произведение 
# векторов a, b в 3D:
def scalarMul3D(a, b):
    # по учебнику из раздела аналитической геометрии
    # a = [x1,y1,z1]
    # b = [x2,y2,z2]
    (x1, y1, z1) = (a[0], a[1], a[2])
    (x2, y2, z2) = (b[0], b[1], b[2])

    return x1*x2 + y1*y2 + z1*z2

# Определим функцию, выполняющую векторное произведение 
# векторов a, b в 3D:
def vectorMul3D(a, b):
    # по учебнику из раздела аналитической геометрии
    # a = [x1,y1,z1]
    # b = [x2,y2,z2]
    (x1, y1, z1) = (a[0], a[1], a[2])
    (x2, y2, z2) = (b[0], b[1], b[2])

    return [y1*z2 - z1*y2, -x1*z2 + z1*x2, x1*y2 - y1*x2]

# Введем класс для работы с прямой в 3D:
# x = P0 + V0*t
# y = P1 + V1*t
# z = P2 + V2*t
# --- т.е. прямая задается начальной координатой P и вектором направления V
class Line3D:
    def __init__(self, a, b): # конструируется (уравнение Ax+By+Cz+D=0) по трем точкам
        self.V = vector3D(a, b)
        self.P = [a[0], a[1], a[2]]

    def point(self, t):
        return [
            self.P[0] + self.V[0]*t,
            self.P[1] + self.V[1]*t,
            self.P[2] + self.V[2]*t
        ]

# Введем класс для работы с плоскостью (все, что нам нужно от 
# полскости --- это находить точку, в которой она пересекается с 3D прямой)
class Plane3D:
    def __init__(self, a, b, c): # конструируется (уравнение Ax+By+Cz+D=0) по трем точкам
        abVector = vector3D(a, b)
        acVector = vector3D(a, c)

        self.ABC = vectorMul3D(abVector, acVector) # вектор нормали плоскости
        self.D   = -(self.ABC[0]*a[0] + self.ABC[1]*a[1] + self.ABC[2]*a[2])

    # находит точку пересечения себя с прямой l
    def intersectionPointWithLine(self, l:Line3D): 
        scalar = scalarMul3D(self.ABC, l.V)
        if (scalar == 0):
            raise ValueError("Прямая и плоскость не пересекаются!")
        
        t = -(scalarMul3D(self.ABC, l.P) + self.D)/scalar

        return l.point(t)


# Определим новые точки C и G (amesC, amesG), отодвинув их на луче зрения.
# Напомним, что луч зрения исходит из точки V.
# Масштаб искажения:
SCALE = 2

amesC = Line3D(V, C).point(SCALE)
amesG = Line3D(V, G).point(SCALE)

# По построению точки B,D,F,H остаются неизменными:
amesB = B
amesD = D
amesF = F
amesH = H

# Теперь можно определить плоскости пола, потолка, правой и дальней стены
# комнаты Эймса:
amesFloor       = Plane3D(amesF, amesH, amesG)
amesCeil        = Plane3D(amesB, amesD, amesC)
amesRightWall   = Plane3D(amesD, amesH, amesC)
amesFrontWall   = Plane3D(amesB, amesF, amesC)

# Осталось найти плоскость левой стены. Луч зрения VA пересечет потолок 
# в точке A', которая также принадлежит левой стене
amesA = amesCeil.intersectionPointWithLine(Line3D(V,A))

# Таким образом, плоскость левой стены тоже определена:
amesLeftWall   = Plane3D(amesA, amesB, amesF)

# Определим функцию, которая строит проекцию точки, на которую смотрят
# из точки V на заданную плоскость:
def viewProjectionPointOnPlane(point, plane:Plane3D):
    return plane.intersectionPointWithLine(Line3D(V, point))

# Определим также функцию, которая проецирует целый массив узоров
def patternsProjection(patterns:list[Pattern], plane:Plane3D):
    projections = []
    for pattern in patterns:
        points = []
        for point in pattern.points:
            points.append(viewProjectionPointOnPlane(point, plane)) # спроецировали каждую точку
        projections.append(Pattern(pattern.picturePath, points))
    return projections

# Находим проекции узоров
amesFloorPatterns       = patternsProjection(floorPatterns, amesFloor)
amesCeilPatterns        = patternsProjection(ceilPatterns, amesCeil)
amesLeftWallPatterns    = patternsProjection(leftWallPatterns, amesLeftWall)
amesRightWallPatterns   = patternsProjection(rightWallPatterns, amesRightWall)
amesFrontWallPatterns   = patternsProjection(frontWallPatterns, amesFrontWall)

# Теперь проекции узоров нужно повернуть так, чтобы они "легли" например, на 
# плоскость xOy (все узоры лежат в одной плоскости по определению). 
# Делать это можно с помощью матрицы поворотов. Поворачивать придется 
# дважды: 
#  * сначала вокруг оси Oz, на угол между ортом Ox и линией пересечения 
#    плоскости с xOy
#  * затем, после предыдущего поворота, вокруг оси Ox, на угол между ортом Oy 
#    и линией пересечения плоскости с yOz
# После этого, координаты z должны стать одинаковыми

# Определим функцию умножения вектора на матрицу:
def vectorMulMatrix3D(v, m):
    r = [0,0,0]
    for i in range(3):
        for j in range(3):
            r[i] += v[j] * m[j][i]
    return r

# Определим функцию умножения матрицу на матрицу
def matrixMulMatrix3D(m1,m2):
    r = [[0,0,0],
         [0,0,0],
         [0,0,0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r[i][j] += m1[i][k]*m2[k][j]
    return r

# достанем функцию квадратного корня из пакета math
from math import sqrt

# Определим функию поворота произвольно расположенной плоскости 
# так, чтобы она стала параллельной плоскости xOy. 
# Поворот сохраняет расстояние между точками, соответственно
# такое преобразование позволяет получить 2D выкройку на плоскости. 
# Выполняется два поворота: первый --- вокруг оси Oz, 
# второй --- вокруг Oy 
# ---
# Функция возвращает результирующую матрицу, на которую можно 
# умножать любую 3D точку, лежащую в исходной плоскости и таким образом 
# получать ее координаты на плоскости

def matrixFor2D_xOy(plane:Plane3D):
    # находим синус и косинус угла поворота вектора нормали плоскости вокруг оси
    # Oz (после поворота компонента y (plane.ABC[1]) вектора нормали должна стать 
    # нулевой)
    s = plane.ABC[1]/sqrt(plane.ABC[0]*plane.ABC[0] + plane.ABC[1]*plane.ABC[1])
    c = plane.ABC[0]/sqrt(plane.ABC[0]*plane.ABC[0] + plane.ABC[1]*plane.ABC[1])

    rm1 = [ # так выглядит матрица поворота вокруг Oz
        [ c,-s, 0],
        [ s, c, 0],
        [ 0, 0, 1]
    ]

    rm1_rev = [ # так выглядит матрица обратного поворота вокруг Oz
                # в математике это обратная матрица. rm1*rm1_rev = E, где
                # E - единичная матрица, умножение на которую ничего не меняет
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1]
    ]

    # повернули исходный вектор нормали (y - составляющая == const)
    newABC = vectorMulMatrix3D(plane.ABC, rm1)

    # синус и косинус угла поворота вокруг Oy
    s = newABC[0]/sqrt(newABC[0]*newABC[0] + newABC[2]*newABC[2])
    c = newABC[2]/sqrt(newABC[0]*newABC[0] + newABC[2]*newABC[2])

    rm2 = [ # матрица поворота вокруг Oy
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ]

    rm2_rev = [ # матрица обратног поворота вокруг Oy
        [ c, 0,-s],
        [ 0, 1, 0],
        [ s, 0, c]
    ]

    # Умножение матриц ассоциативно, но некоммутативно. Поэтому,
    # перемножая матрицы, получим сразу матрицу двух поворотов!!!
    # (v*rm1)*rm2 == v*(rm1*rm2)
    # M = rm1*rm2

    # также вернем матирцу обратного двойного поворота: 
    # M_rev = rm2_rev*rm1_rev
    # т.к.: rm1*rm1_rev = E
    #    и: rm2*rm2_rev = E
    # То:
    #   M*M_rev = (rm1*rm2)*(rm2_rev*rm1_rev) = rm1*rm2*rm2_rev*rm1_rev =
    #   = rm1*(rm2*rm2_rev)*rm1_rev = rm1*E*rm1_rev = rm1*rm1_rev = E
    return (matrixMulMatrix3D(rm1, rm2), matrixMulMatrix3D(rm2_rev, rm1_rev))
                                        

# Определим функцию, которая преобразует точки массива узоров, 
# умножая их на матрицу m (с помощью матриц можно, например,
# вращать и отражать точки(вектора))
def patternsMulMatrix3D(patterns:list[Pattern], m):
    newPatterns = []
    for pattern in patterns:
        points = []
        for point in pattern.points:
            points.append(vectorMulMatrix3D(point, m)) # умножаем каждую точку
        newPatterns.append(Pattern(pattern.color, points))
    return newPatterns

# Определим функцию, которая поворачивает массив узоров, которые лежат в
# одной плоскости (аргумент plane)
def patternsRotationFor2D_xOy(patterns:list[Pattern], plane:Plane3D):
    return patternsMulMatrix3D(patterns, matrixFor2D_xOy(plane))

# получили выкройки на плоскости (координаты z у всех точек выкройки для каждого 
# узора в списке --- одинаковые)
ames2D_xOy_FloorPatterns       = patternsRotationFor2D_xOy(amesFloorPatterns, amesFloor)
ames2D_xOy_CeilPatterns        = patternsRotationFor2D_xOy(amesCeilPatterns, amesCeil)
ames2D_xOy_LeftWallPatterns    = patternsRotationFor2D_xOy(amesLeftWallPatterns, amesLeftWall)
ames2D_xOy_RightWallPatterns   = patternsRotationFor2D_xOy(amesRightWallPatterns, amesRightWall)
ames2D_xOy_FrontWallPatterns   = patternsRotationFor2D_xOy(amesFrontWallPatterns, amesFrontWall)

# сделаем также выкройки базовой комнаты (все, кроме дальней стены, которая и 
# xOy, нужно будет повернуть):
room2D_xOy_FloorPatterns       = patternsRotationFor2D_xOy(floorPatterns, Plane3D(E, F, G))
room2D_xOy_CeilPatterns        = patternsRotationFor2D_xOy(ceilPatterns, Plane3D(A, B, D))
room2D_xOy_LeftWallPatterns    = patternsRotationFor2D_xOy(leftWallPatterns, Plane3D(E, A, F))
room2D_xOy_RightWallPatterns   = patternsRotationFor2D_xOy(rightWallPatterns, Plane3D(H, D, G))
room2D_xOy_FrontWallPatterns   = frontWallPatterns

# Определим функцию, возвращающую параметры прямогугольной области
# (xMin, yMin, width, height), которая содержит все узоры в заданном массиве узоров
# Функция работает с выкройкой (т.е. использует только координаты x,y трехмерной точки
# [x,y,z])
def patternsArea2D_xOy(patterns: list[Pattern]):
    if (len(patterns) == 0):
        raise ValueError("Пустые массивы узоров не принимаются!")

    xMin = xMax = patterns[0].points[0][0]
    yMin = yMax = patterns[0].points[0][1]

    for pattern in patterns:
        for point in pattern.points:
            x = point[0]
            if x < xMin: xMin = x
            if x > xMax: xMax = x

            y = point[1]
            if y < yMin: yMin = y
            if y > yMax: yMax = y
    
    return (xMin, yMin, xMax - xMin, yMax - yMin)


# Так как мы формировали узор в комнате, как бы смотря "изнутри", то 
# получается, что на проекции (которая по вектору одного направления) 
# мы как бы смотрим на узор стены "снаружи" комнаты). 
# Поэтому часть узоров нужно печатать в отражении.
# Поэтому определим несколько зеркально "отражащюих" функций:

# отражение по xOz (y = -y)
def mirrorOx(patterns: list[Pattern]):
    m = [   [1, 0, 0], # это матрица отражения y = -y
            [0,-1, 0],
            [0, 0, 1] ]
    return patternsMulMatrix3D(patterns, m)

# отражение по Oy (x = -x)
def mirrorOy(patterns: list[Pattern]):
    m = [   [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1] ]
    return patternsMulMatrix3D(patterns, m)

# точки узора должны образовывать выпуклый многоугольник
def isPointInPattern_xOy(point: list[float], pattern: Pattern):
    pointsCount = len(pattern.points)
    if pointsCount < 3:
        return False

    i = 0
    while i < pointsCount:
        
        # выберем две точки i-й стороны
        (x1, y1) = (pattern.points[i][0], pattern.points[i][1])

        (x2, y2) = (pattern.points[0][0], pattern.points[0][1])
        if i + 1 < pointsCount:
            (x2, y2) = (pattern.points[i+1][0], pattern.points[i+1][1])

        # быстренько определим критерий проверки, чтобы не писать лишнего
        side = lambda x, y: (x-x1)*(y2-y1) - (y-y1)*(x2-x1)

        # проверим, что все остальные точки узора, а также проверяемая точка 
        # находятся с одной стороны относительно i-й стороны ((x1,y1), (x2,y2))
        sidePos = sideNeg = 0
        j = 0
        while j < pointsCount:
            if not ( (j == i) or (j == i + 1) or ((i + 1 >= pointsCount) and j == 0) ):
                sideVal = side(pattern.points[j][0], pattern.points[j][1])
                if sideVal < 0: 
                    sideNeg = sideNeg + 1
                elif sideVal > 0:
                    sidePos = sidePos + 1
            j = j + 1

        sideVal = side(point[0], point[1])
        if sideVal < 0: 
            sideNeg = sideNeg + 1
        elif sideVal > 0:
            sidePos = sidePos + 1

        if (sideNeg != 0) and (sidePos != 0):
            return False

        i = i + 1

    return True

print("Done. Use images files in the 'output' directory")
