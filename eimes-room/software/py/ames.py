# Это программа-генератор выкроек стен, потолка и пола комнаты Эймса. 
# Выкройки будут генерироваться в векторном формате (SVG или METAPOST)
# Бери, печатай, вырезай, клей.

# Обозначения элементов комнаты Эймса 
# возьмем как на рисунке:
# "../../../eimes-room/beamer/figs/ames-only.png"
#
# Т.е.:
# * floor       - пол (EFGH);
# * ceil        - потолок (ABCD);
# * leftWall    - левая стена (EABF);
# * frontWall   - передняя стена (FBCG);
# * rightWall   - правая стена (GCDH);
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

# На стенах, полу и потолке комнаты могут быть линейные узоры.
# Элемент узора (например, одна кафельная плитка на полу) в этой программе 
# представляет собой замкнутую фигуру из нескольких 2D или 3D точек (хранимых списком), 
# лежащих в одной плоскости, залитую одним цветом:

from dataclasses import dataclass

@dataclass
class Pattern:
    color: str
    points: list[list[float]]

# Так как каждый элемент узора (все его точки) будет испытывать одни и те же
# линейные преобразования, то будем обрабатывать сразу массивы таких элементов.

# Определим функции для генерации узоров комнаты:

def generatePatternsChessFloor():
    print("calkjlk lkjlkjlkj")
    return []

def generatePatternsCeil():
    pass #TODO

def generatePatternsLeftWall():
    pass #TODO

def generatePatternsRightWall():
    pass #TODO

def generatePatternsFrontWall():
    pass #TODO


floorPatterns       = generatePatternsChessFloor() #TODO
ceilPatterns        = generatePatternsCeil() #TODO

leftWallPatterns    = generatePatternsLeftWall() #TODO
rightWallPatterns   = generatePatternsRightWall() #TODO
frontWallPatterns   = generatePatternsFrontWall() #TODO



# Функции для генерации узоров




