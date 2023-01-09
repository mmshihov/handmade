# Это программа-генератор выкроек стен, потолка и пола комнаты Эймса. 
# Выкройки будут генерироваться в векторном формате (SVG или METAPOST)
# Бери, печатай, вырезай, клей.

# Обозначения элементов комнаты Эймса 
# возьмем как на рисунке:
# "../../../eimes-room/beamer/figs/ames-only.png" 
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

# Определим несколько функций для генерации узоров комнаты:

# создаем пол в шахматную клеточку (выложен плиткой)
def generatePatternsChessFloor(): 
    FIELD_COUNT_ON_WIDTH = 10
    COLORS = ["black", "white"]

    w = (G[0] - F[0]) / FIELD_COUNT_ON_WIDTH # плитка квадратная, размером w*w
        
    x = F[0] # начинаем генерировать клеточки с левого-нижнего-дальнего угла
    z = F[2] # x, z --- координаты одного из уголоков плитки
    ix = iz = 0 # номер плитки по ширине и глубине (двумерный массив плиток)
                # для определения того, каким цветом красить плитку 
                # в шахматку ((ix+iy) mod COLORS.COUNT)
    patterns = [Pattern("white", [E, F, G, H])] # первым элементом списка будет весь пол
    # а только потом --- плитки:
    while (x < G[0]):
        z = F[2]
        iz = 0
        while (z > E[2]):
            # может быть последние плитки будут не целыми, поэтому определяем 
            # ширину и глубину конкретной плитки:
            wx = wz = w
            if (x + wx > G[0]):
                wx = G[0] - x
            if (z - wz < E[2]):
                wz = z - E[2]

            color = COLORS[(ix + iz) % len(COLORS)] # определяем цвет добавляемой плитки

            # добавляем плитку в масссив узоров
            patterns.append(Pattern( # создаем узор плитки, указывая:
                color,  # цвет плитки
                [       # координаты точек плитки, лежащей на полу (плоскость xOz):
                    [x,         E[1],   z],
                    [x + wx,    E[1],   z],
                    [x + wx,    E[1],   z - wz],
                    [x,         E[1],   z - wz]
                ]
            ))

            z = z - w
            iz = iz + 1
        x = x + w
        ix = ix + 1

    return patterns

# На потолке будет пять квадратных светильников (четыре по краям, один большой в центре)
def generatePatternsCeil():
    LIGHT_DIMENSION_PER_WIDTH = 10 # сколько светильников помещается по ширине (минимум)
    COLOR = "blue"

    wx = (C[0] - B[0]) / LIGHT_DIMENSION_PER_WIDTH # размер светильника по ширине
    wz = (B[2] - A[2]) / LIGHT_DIMENSION_PER_WIDTH # размер светильника по глубине

    w = wx
    if (wz < w): w = wz

    patterns = [Pattern("white", [A, B, C, D])] # первым элементом списка будет весь потолок

    x = B[0] + w;  z = B[2] - 2*w
    patterns.append(Pattern(
        COLOR,
        [
            [x,     A[1], z],
            [x + w, A[1], z],
            [x + w, A[1], z + w],
            [x,     A[1], z + w]
        ]))

    # светильники:
    x = B[0] + w;  z = B[2] - 2*w       # левый-дальний
    patterns.append(Pattern(COLOR,[ [x, A[1], z], [x + w, A[1], z], [x + w, A[1], z + w], [x, A[1], z + w]]))

    x = B[0] + w;  z = A[2] + w         # левый-ближний
    patterns.append(Pattern(COLOR,[ [x, A[1], z], [x + w, A[1], z], [x + w, A[1], z + w], [x, A[1], z + w]]))

    x = C[0] - 2*w;  z = C[2] - 2*w     # правый-дальний
    patterns.append(Pattern(COLOR,[ [x, A[1], z], [x + w, A[1], z], [x + w, A[1], z + w], [x, A[1], z + w]]))

    x = C[0] - 2*w;  z = D[2] + w       # правый-ближний
    patterns.append(Pattern(COLOR,[ [x, A[1], z], [x + w, A[1], z], [x + w, A[1], z + w], [x, A[1], z + w]]))

    # центральный (больше остальных в два раза линейно)
    x = (B[0] + C[0]) / 2;  z = (A[2] + B[2]) / 2
    patterns.append(Pattern(
        COLOR,
        [
            [x - w, A[1], z - w],
            [x - w, A[1], z + w],
            [x + w, A[1], z + w],
            [x + w, A[1], z - w],
        ]))

    return patterns

# На всех стенах есть фартук:
BORDER_HEIGHT = HEIGHT / 2
BORDER_COLOR  = "green"

# левая стена с дверью
def generatePatternsLeftWall():
    DOOR_HEIGHT = HEIGHT * 4 / 5
    DOOR_WIDTH  = DOOR_HEIGHT / 2.5
    DOOR_COLOR = "grey"

    patterns = [Pattern("white", [E, A, B, F])] # стена целиком

    # фартук
    patterns.append(Pattern(
        BORDER_COLOR,
        [
            E,
            [E[0], E[1] + BORDER_HEIGHT, E[2]],
            [F[0], F[1] + BORDER_HEIGHT, F[2]],
            F
        ]))
    
    # дверь


# правая стена с окном или картиной
def generatePatternsRightWall():
    pass #TODO

# дальняя стена с камином или панно
def generatePatternsFrontWall():
    pass #TODO


floorPatterns       = generatePatternsChessFloor()
print(floorPatterns)

ceilPatterns        = generatePatternsCeil() #TODO

leftWallPatterns    = generatePatternsLeftWall() #TODO
rightWallPatterns   = generatePatternsRightWall() #TODO
frontWallPatterns   = generatePatternsFrontWall() #TODO



# Функции для генерации узоров




