import os
from math import floor, sqrt
from PIL import Image

# Это программа-генератор искаженных в линейной перспективе картинок

# Трехментые координаты будут представлены списком из трех чисел.
# Двумерные (для выкроек стен, потолка и пола комнаты) --- списком из двух чисел.
# 3D точка: [x, y, z], 2D: [x, y].
# Измерения: x - ширина (право-лево), y - высота (верх-низ), z - глубина (вблизи-вдали)
# Измеряем всё в чем попало, например в пикселях!

# Элемент узора (на который будет "натягиваться" картинка) в этой программе 
# представляет собой замкнутую фигуру из нескольких 2D или 3D точек (хранимых списком), 
# лежащих в одной плоскости:

class Pattern:
    # picturePath - путь к картинке
    # points - точки
    def __init__(self, picturePath: str, points: list[list[float]]):
        self.picturePath = picturePath
        self.points = points


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


# Определим функцию, которая строит проекцию точки, на которую смотрят
# из точки V на заданную плоскость:
def viewProjectionPointOnPlane(point, plane:Plane3D):
    return plane.intersectionPointWithLine(Line3D(V, point))

# Определим функцию, которая проецирует узор на плоскость через точку зрения V
def patternProjection(pattern: Pattern, plane:Plane3D):
    newPicturePath = os.path.join(
        "output/pictures", 
        "scale_" + os.path.basename(pattern.picturePath))

    points = []
    for point in pattern.points:
        points.append(viewProjectionPointOnPlane(point, plane)) # спроецировали каждую точку

    return Pattern(newPicturePath, points)


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

def matrixFor_XY(plane:Plane3D):
    # находим синус и косинус угла поворота вектора нормали плоскости вокруг оси
    # Oz (после поворота компонента y (plane.ABC[1]) вектора нормали должна стать 
    # нулевой)
    s = 0
    c = 1
    d = sqrt(plane.ABC[0]*plane.ABC[0] + plane.ABC[1]*plane.ABC[1])
    if d != 0:
        s = plane.ABC[1]/d
        c = plane.ABC[0]/d

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
    s = 0
    c = 1
    d = sqrt(newABC[0]*newABC[0] + newABC[2]*newABC[2])
    if (d != 0):
        s = newABC[0]/d
        c = newABC[2]/d

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
                                        

# Определим функцию, которая преобразует точки узора,
# умножая каждую на матрицу m
def patternMulMatrix3D(pattern: Pattern, m):
    points = []
    for point in pattern.points:
        points.append(vectorMulMatrix3D(point, m))
    return Pattern(pattern.picturePath, points)


# Определим функцию, возвращающую параметры прямогугольной области
# (xMin, yMin, width, height), которая содержит узор.
# Функция работает с выкройкой (т.е. использует только координаты 
# x,y трехмерной точки [x,y,z])
def patternArea_XY(pattern: Pattern):
    xMin = xMax = pattern.points[0][0]
    yMin = yMax = pattern.points[0][1]

    for point in pattern.points:
        x = point[0]
        if x < xMin: xMin = x
        if x > xMax: xMax = x

        y = point[1]
        if y < yMin: yMin = y
        if y > yMax: yMax = y
    
    return (xMin, yMin, xMax - xMin, yMax - yMin)


# точки узора должны образовывать выпуклый многоугольник
def isPointInPattern_XY(point: list[float], pattern: Pattern):
    pointsCount = len(pattern.points)
    if pointsCount < 3:
        return False

    i = 0
    while i < pointsCount:
        # выберем две точки i-й стороны: (i, i+1)
        (x1, y1) = (pattern.points[i][0], pattern.points[i][1])

        (x2, y2) = (pattern.points[0][0], pattern.points[0][1])
        if i + 1 < pointsCount:
            (x2, y2) = (pattern.points[i+1][0], pattern.points[i+1][1])

        # критерий проверки, чтобы не писать лишнего
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


# Функция, которая картинку базового узора (basePattern) "натягивает" 
# на его проекцию на заданную плоскость amesPlane.
# Базовый узор  должен лежать в плоскости basePlane (не проверяем это!).
# brmXY, armXY --- матрицы ОТРАЖЕНИЯ чтобы контролировать результат
def makePicture(basePattern: Pattern, brmXY, basePlane, armXY, amesPlane):
    amesPattern = patternProjection(basePattern, amesPlane)

    mBase,_ = matrixFor_XY(basePlane)
    mBase   = matrixMulMatrix3D(mBase, brmXY)
    basePatternXY = patternMulMatrix3D(basePattern, mBase)
    baseX, baseY, baseLenX, baseLenY = patternArea_XY(basePatternXY)

    # используем библтотеку Pillow для работы с изображениями
    baseIm = Image.open(basePattern.picturePath)
    (basePixelCountX, basePixelCountY) = baseIm.size

    mAmes, mAmes_rev = matrixFor_XY(amesPlane)
    mAmes, mAmes_rev = matrixMulMatrix3D(mAmes, armXY), matrixMulMatrix3D(armXY, mAmes_rev)
    amesPatternXY = patternMulMatrix3D(amesPattern, mAmes)
    amesX, amesY, amesLenX, amesLenY = patternArea_XY(amesPatternXY)
    amesZ = amesPatternXY.points[0][2]  # z координата (с индексом 2 в массиве) 
                                        # у всех точек паттерна должна быть одинакова

    # используем ту же плотность пикселей, что и в базовой картинке
    amesPixelCountX = floor((amesLenX * basePixelCountX) / baseLenX)
    amesPixelCountY = floor((amesLenY * basePixelCountY) / baseLenY)

    # создаем картинку Эймса размерностью (amesPixelCountX, amesPixelCountY)
    amesIm = Image.new(baseIm.mode, (amesPixelCountX, amesPixelCountY))

    # проходим по всем пикселям картинки Эймса
    amesPixelX = 0
    while amesPixelX < amesPixelCountX:
        amesPixelY = 0
        while amesPixelY < amesPixelCountY:
            # масштабируем пиксел в координату на плоском узоре Эймса
            amesPointXY = [
                amesX + amesPixelX * amesLenX / amesPixelCountX,
                amesY + amesPixelY * amesLenY / amesPixelCountY,
                amesZ
            ]

            # проверяем попала ли точка в узор, чтобы не делать лишних вычислений:
            if isPointInPattern_XY(amesPointXY, amesPatternXY):
                # поворачиваем на исходную плоскость Эймса
                amesPoint = vectorMulMatrix3D(amesPointXY, mAmes_rev)
                basePoint = viewProjectionPointOnPlane(amesPoint, basePlane)
                basePointXY = vectorMulMatrix3D(basePoint, mBase)

                # возможно не попали внутрь базового узора:
                if isPointInPattern_XY(basePointXY, basePatternXY):
                    basePixelX = floor((basePointXY[0] - baseX) * basePixelCountX / baseLenX)
                    basePixelY = floor((basePointXY[1] - baseY) * basePixelCountY / baseLenY)

                    # пишем пиксел (basePixelX, basePixelY) исходной картинки в пиксел (amesPixelX, amesPixelY) Эймса
                    if (basePixelX < basePixelCountX) and (basePixelY < basePixelCountY):
                        amesIm.putpixel((amesPixelX, amesPixelY), baseIm.getpixel((basePixelX, basePixelY)))

            # print empty pixel to (amesPixelX, amesPixelY)

            amesPixelY = amesPixelY + 1
        amesPixelX = amesPixelX + 1
    
    amesIm.save(amesPattern.picturePath, baseIm.format, dpi=baseIm.info["dpi"])


# Точку зрения V поместим в начало координат: 
V = [0, 0, 0]

# расстояние глаза от картинки около полуметра (иначе глаз не ловит фокус)
distance = 5*480

# Зададим (прямоугольный) узор, содержащий картинку:
pattern = Pattern(
    "input/pictures/stitch.png",
    [
        [-160, -240, distance],
        [-160,  240, distance],
        [ 160,  240, distance],
        [ 160, -240, distance],        
    ])

projectionPlane = Plane3D(
    [-160, -240, distance],
    [ 160, -240, distance],
    [-160,  240, distance + 2*480],
)

# Находим проекцию узора
e =   [ [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1] ]

mx = [  [-1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1] ]

my = [  [ 1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1] ]

mxy = [ [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1] ]

makePicture(
    pattern, 
    my, 
    Plane3D(pattern.points[0], pattern.points[1], pattern.points[2]), 
    mxy, 
    projectionPlane)

print("Done. Use image files in the 'output' directory")
