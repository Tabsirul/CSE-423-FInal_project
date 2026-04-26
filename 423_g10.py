from OpenGL.GL   import *
from OpenGL.GLUT import *
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18
from OpenGL.GLU  import *
import math, random, time, heapq, sys

WINDOW_W  = 1000
WINDOW_H  = 800
CELL_SIZE = 80
MAZE_SIZE = 15
HALF_GRID = (MAZE_SIZE * CELL_SIZE) // 2
PIT_DEPTH      = 300.0
SPIKE_HEIGHT   = 55.0
SPIKE_BASE     = 7.0
camera_x = 0
camera_y = 650
camera_z = 700
player_pos        = [0.0, 0.0, 10.0]
player_angle      = 270.0
player_speed      = 340.0          
player_start_pos  = [0.0, 0.0, 10.0]
fp_pitch          = 0.0
score                = 0
lives                = 5
cheat_mode           = False
first_person         = False
game_over            = False
diamond_found        = False
game_start_time      = 0.0
game_time            = 180
AUTO_FP_DELAY        = 3
hit_invincible_until = 0.0
HIT_INVINCIBLE_SECS  = 1.5
_last_idle_time = 0.0
dt              = 0.016
HOLE_COUNT           = 6
hole_cells           = []          
_all_hole_cells      = []          
HOLE_VISIBLE_SECS    = 4.0         
HOLE_HIDDEN_SECS     = 6.0        
_hole_phase_start    = 0.0         
_holes_visible       = False       
falling          = False
fall_z           = 10.0
FALL_SPEED       = 180.0   
RESPAWN_WAIT     = 3.0      
fall_start_time  = 0.0
fall_px          = 0.0     
fall_py          = 0.0
enemies              = []
ENEMY_COUNT          = 4
ENEMY_BASE_SPEED     = 80.0
REPATH_INTERVAL      = 0.5
_enemy_boost_until   = 0.0
ENEMY_BOOST_DURATION = 5.0
bullets        = []
BULLET_SPEED   = 420.0
BULLET_RADIUS  = 6.0
BULLET_RANGE   = 750.0
SHOT_COOLDOWN  = 0.35
last_shot_time = 0.0
big_ball_pos            = [0.0, 0.0, 50.0]
BIG_BALL_RADIUS         = 28.0
BIG_BALL_SPEED          = 18.0
BIG_BALL_SPAWN_DELAY    = 12.0
big_ball_active         = False
big_ball_spawned        = False
big_ball_respawn_timer  = 0.0
BIG_BALL_RESPAWN_DELAY  = 5.0
BIG_BALL_MIN_SPAWN_DIST = 380.0
diamond_pos = [0.0, 0.0, 1.0]
diamond_row = 0
diamond_col = 0
question_active     = False
current_question    = None
questions_answered  = 0
MAX_QUESTIONS       = 5
remaining_questions = []
show_map_view       = False
map_view_start      = 0.0
MAP_VIEW_DURATION   = 4.0

QUESTION_BANK = [
    {"q": "What does CPU stand for?",
     "opts": ["Central Processing Unit", "Core Power Unit"], "ans": 0},
    {"q": "Which language runs natively in browsers?",
     "opts": ["Python", "JavaScript"],                       "ans": 1},
    {"q": "What is 7 x 8?",
     "opts": ["54", "56"],                                   "ans": 1},
    {"q": "Who invented the World Wide Web?",
     "opts": ["Tim Berners-Lee", "Bill Gates"],              "ans": 0},
    {"q": "Which data structure follows LIFO?",
     "opts": ["Queue", "Stack"],                             "ans": 1},
]

#  MAZE GENERATION (Member 3 –recursive DFS)
def build_maze(n):
    grid = [['1'] * n for _ in range(n)]
    def carve(cx, cy):
        dirs = [(0,2),(0,-2),(2,0),(-2,0)]
        random.shuffle(dirs)
        for ddx, ddy in dirs:
            nx, ny = cx+ddx, cy+ddy
            if 1 <= nx < n-1 and 1 <= ny < n-1 and grid[ny][nx] == '1':
                grid[ny - ddy//2][nx - ddx//2] = '0'
                grid[ny][nx] = '0'
                carve(nx, ny)
    grid[1][1] = '0'
    carve(1,1)
    grid[0][1]     = '0'
    grid[n-1][n-2] = '0'
    return ["".join(row) for row in grid]

maze_grid = build_maze(MAZE_SIZE)

# member 3
#  COORDINATE HELPERS
def cell_to_world(row, col):
    wx = col * CELL_SIZE - HALF_GRID + CELL_SIZE // 2
    wy = (MAZE_SIZE - 1 - row) * CELL_SIZE - HALF_GRID + CELL_SIZE // 2
    return float(wx), float(wy)

def world_to_cell(wx, wy):
    col = int((wx + HALF_GRID) // CELL_SIZE)
    row = int(MAZE_SIZE - 1 - (wy + HALF_GRID) // CELL_SIZE)
    return (max(0, min(MAZE_SIZE-1, row)), max(0, min(MAZE_SIZE-1, col)))

# member 1 
def touches_wall(wx, wy, probe=12):
    for ox in (-probe, 0, probe):
        for oy in (-probe, 0, probe):
            r, c = world_to_cell(wx+ox, wy+oy)
            if maze_grid[r][c] == '1':
                return True
    return False

#  A* PATHFINDING  (Member 1: Nafisa Tabassum)
def astar_path(start_rc, goal_rc):
    sr, sc = start_rc
    gr, gc = goal_rc
    if maze_grid[sr][sc] == '1' or maze_grid[gr][gc] == '1':
        return []
    if (sr, sc) == (gr, gc):
        return []
    def h(r, c): return abs(r-gr) + abs(c-gc)
    heap = []
    heapq.heappush(heap, (h(sr,sc), 0, sr, sc))
    g_cost    = {(sr,sc): 0}
    came_from = {}
    while heap:
        _, g, r, c = heapq.heappop(heap)
        if (r,c) == (gr,gc):
            path, node = [], (r,c)
            while node in came_from:
                path.append(node); node = came_from[node]
            path.reverse(); return path
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<MAZE_SIZE and 0<=nc<MAZE_SIZE and maze_grid[nr][nc]=='0':
                ng = g+1
                if ng < g_cost.get((nr,nc), 10**9):
                    g_cost[(nr,nc)]    = ng
                    came_from[(nr,nc)] = (r,c)
                    heapq.heappush(heap, (ng+h(nr,nc), ng, nr, nc))
    return []

# member 2 
def place_holes():
    global hole_cells, _all_hole_cells, _hole_phase_start, _holes_visible
    _all_hole_cells = []
    hole_cells      = []
    protected = set()
    for r in range(3):
        for c in range(MAZE_SIZE):
            protected.add((r, c))
    protected.add((MAZE_SIZE-1, MAZE_SIZE-2))
    attempts = 0
    while len(_all_hole_cells) < HOLE_COUNT and attempts < 1000:
        attempts += 1
        r = random.randint(3, MAZE_SIZE-2)
        c = random.randint(1, MAZE_SIZE-2)
        key = (r, c)
        if maze_grid[r][c] == '0' and key not in protected and key not in _all_hole_cells:
            _all_hole_cells.append(key)
    # Start hidden  then holes  will appear after HOLE_HIDDEN_SECS
    _holes_visible    = False
    _hole_phase_start = time.time()

# member 1
def push_enemies_from_spawn():
    px, py = player_start_pos[0], player_start_pos[1]
    for e in enemies:
        if not e['active']: continue
        dist = math.hypot(e['pos'][0]-px, e['pos'][1]-py)
        if dist < 180:
            ang = math.atan2(e['pos'][1]-py, e['pos'][0]-px) if dist>1 else random.uniform(0,2*math.pi)
            e['pos'][0] = px + 220*math.cos(ang)
            e['pos'][1] = py + 220*math.sin(ang)
            e['path'] = []

# member 1
def spawn_enemies():
    global enemies
    enemies = []
    start_wx, start_wy = cell_to_world(0,1)
    min_dist = CELL_SIZE*5
    safe_row = MAZE_SIZE//3
    for _ in range(ENEMY_COUNT):
        for _try in range(600):
            r = random.randint(safe_row, MAZE_SIZE-2)
            c = random.randint(1, MAZE_SIZE-2)
            if maze_grid[r][c] != '0': continue
            wx, wy = cell_to_world(r,c)
            if math.hypot(wx-start_wx, wy-start_wy) >= min_dist:
                enemies.append({
                    'pos':         [wx, wy, 20.0],
                    'active':      True,
                    'scale':       1.0,
                    'scale_dir':   0.01,
                    'speed':       ENEMY_BASE_SPEED,
                    'path':        [],
                    'last_repath': 0.0,
                })
                break

def reset_game():
    global player_pos, player_angle, fp_pitch, player_start_pos
    global diamond_pos, diamond_row, diamond_col
    global score, lives, cheat_mode, first_person, game_over, diamond_found
    global game_start_time, game_time, bullets, last_shot_time
    global falling, fall_z, fall_start_time, fall_px, fall_py
    global question_active, current_question, questions_answered, remaining_questions
    global show_map_view, map_view_start
    global big_ball_pos, big_ball_active, big_ball_spawned, big_ball_respawn_timer
    global hit_invincible_until, _enemy_boost_until
    global _last_idle_time, dt, maze_grid
    global camera_x, camera_y, camera_z
    global hole_cells, _all_hole_cells, _hole_phase_start, _holes_visible

    maze_grid = build_maze(MAZE_SIZE)

    score                = 0
    lives                = 5
    cheat_mode           = False
    first_person         = False
    fp_pitch             = 0.0
    game_over            = False
    diamond_found        = False
    game_time            = 180
    bullets              = []
    last_shot_time       = 0.0
    falling              = False
    fall_z               = 10.0
    fall_start_time      = 0.0
    fall_px              = 0.0
    fall_py              = 0.0
    question_active      = False
    current_question     = None
    questions_answered   = 0
    remaining_questions  = list(QUESTION_BANK)
    show_map_view        = False
    map_view_start       = 0.0
    hit_invincible_until = 0.0
    _enemy_boost_until   = 0.0
    camera_x, camera_y, camera_z = 0, 650, 700

    wx, wy = cell_to_world(0,1)
    player_pos[:]    = [wx, wy, 10.0]
    player_start_pos = list(player_pos)
    player_angle     = 270.0

    diamond_row, diamond_col = MAZE_SIZE-1, MAZE_SIZE-2
    dx, dy = cell_to_world(diamond_row, diamond_col)
    diamond_pos[:] = [dx, dy, 1.0]

    ang = random.uniform(0, 2*math.pi)
    big_ball_pos[:]        = [wx + BIG_BALL_MIN_SPAWN_DIST*math.cos(ang),
                               wy + BIG_BALL_MIN_SPAWN_DIST*math.sin(ang), 50.0]
    big_ball_active        = False
    big_ball_spawned       = False
    big_ball_respawn_timer = 0.0

    place_holes()
    spawn_enemies()
    now             = time.time()
    game_start_time = now
    _last_idle_time = now
    dt              = 0.016

def draw_text_2d(x, y, text, font=GLUT_BITMAP_HELVETICA_18, color=(1,1,1)):
    glColor3f(*color)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 1000, 0, 800)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_rect_2d(x1, y1, x2, y2, color):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 1000, 0, 800)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex3f(x1,y1,0); glVertex3f(x2,y1,0)
    glVertex3f(x2,y2,0); glVertex3f(x1,y2,0)
    glEnd()
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# member 3
def draw_floor():
    for ri in range(MAZE_SIZE):
        for ci in range(MAZE_SIZE):
            wx, wy = cell_to_world(ri, ci)
            half = CELL_SIZE / 2.0

            if (ri+ci) % 2 == 0:
                glColor3f(0.18, 0.42, 0.18)
            else:
                glColor3f(0.13, 0.30, 0.13)
            glBegin(GL_QUADS)
            glVertex3f(wx-half, wy-half, 0)
            glVertex3f(wx+half, wy-half, 0)
            glVertex3f(wx+half, wy+half, 0)
            glVertex3f(wx-half, wy+half, 0)
            glEnd()

            g = 3.0
            glColor3f(0.04, 0.10, 0.04)
            glBegin(GL_QUADS)
            # bottom strip
            glVertex3f(wx-half,     wy-half,     0.5)
            glVertex3f(wx+half,     wy-half,     0.5)
            glVertex3f(wx+half,     wy-half+g,   0.5)
            glVertex3f(wx-half,     wy-half+g,   0.5)
            glEnd()
            glBegin(GL_QUADS)
            # left strip
            glVertex3f(wx-half,     wy-half,     0.5)
            glVertex3f(wx-half+g,   wy-half,     0.5)
            glVertex3f(wx-half+g,   wy+half,     0.5)
            glVertex3f(wx-half,     wy+half,     0.5)
            glEnd()

# member 3
def draw_maze():
    for ri, row in enumerate(maze_grid):
        for ci, cell in enumerate(row):
            if cell != '1': continue
            wx, wy = cell_to_world(ri, ci)

            # Wall body
            glPushMatrix()
            glTranslatef(wx, wy, 50)
            glScalef(CELL_SIZE, CELL_SIZE, 100)
            shade = 0.30 + 0.25 * ((ri % 3) / 3.0)
            if (ri+ci) % 2 == 0:
                glColor3f(shade*1.1, shade*1.0, shade*1.2)
            else:
                glColor3f(shade*0.85, shade*0.70, shade*0.50)
            glutSolidCube(1)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(wx, wy, 102)
            glScalef(CELL_SIZE, CELL_SIZE, 4)
            glColor3f(0.80, 0.78, 0.90)
            glutSolidCube(1)
            glPopMatrix()

# member 3
def draw_entrance():
    """Glowing arch over the maze entrance at row=0, col=1."""
    t  = time.time()
    wx, wy = cell_to_world(0, 1)
    # Two glowing pillars
    for sx in (-38, 38):
        glPushMatrix()
        glTranslatef(wx+sx, wy+8, 60)
        glScalef(8, 8, 120)
        glow = 0.6 + 0.3*math.sin(t*2.0 + sx)
        glColor3f(0.2, glow, 0.4)
        glutSolidCube(1)
        glPopMatrix()
    # Top lintel sphere row
    for i in range(5):
        glow = 0.5 + 0.4*math.sin(t*3.0 + i*0.9)
        glColor3f(0.1, glow, 0.3)
        glPushMatrix()
        glTranslatef(wx - 24 + i*12, wy+5, 125)
        gluSphere(gluNewQuadric(), 5, 8, 8)
        glPopMatrix()

# member 3
_torch_positions = []   
def build_torch_positions():
    """Pick a subset of open cells adjacent to walls for torch placement."""
    global _torch_positions
    _torch_positions = []
    rng = random.Random(42)         
    count = 0
    for ri in range(1, MAZE_SIZE-1):
        for ci in range(1, MAZE_SIZE-1):
            if maze_grid[ri][ci] == '0' and count < 20:
                has_wall = any(
                    maze_grid[ri+dr][ci+dc] == '1'
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                    if 0<=ri+dr<MAZE_SIZE and 0<=ci+dc<MAZE_SIZE
                )
                if has_wall and rng.random() < 0.22:
                    _torch_positions.append(cell_to_world(ri,ci))
                    count += 1

# member 3
def draw_torches():
    t = time.time()
    for i, (tx, ty) in enumerate(_torch_positions):
        flick  = 0.55 + 0.45 * math.sin(t * 7.3 + i * 1.7)
        flick2 = 0.40 + 0.40 * math.sin(t * 11.1 + i * 2.3)

        glPushMatrix()
        glTranslatef(tx, ty, 45)
        glScalef(3, 3, 18)
        glColor3f(0.30, 0.18, 0.05)
        glutSolidCube(1)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(tx, ty, 58)
        glColor3f(1.0, flick*0.55, 0.0)
        gluSphere(gluNewQuadric(), 5.5, 8, 8)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(tx, ty, 61 + flick*3)
        glColor3f(1.0, flick2*0.9 + 0.1, flick2*0.3)
        gluSphere(gluNewQuadric(), 3.0, 6, 6)
        glPopMatrix()

# member 2 
#  DRAW: SPIKE PIT  (visible inside each hole)
def draw_holes():
    t = time.time()
    pulse = 0.5 + 0.5 * math.sin(t * 4.0)

    for hr, hc in hole_cells:
        wx, wy = cell_to_world(hr, hc)
        half   = CELL_SIZE / 2.0 - 1

        shaft_h = PIT_DEPTH
        glColor3f(0.10, 0.02, 0.02)
        # Bottom of pit
        glPushMatrix()
        glTranslatef(wx, wy, -PIT_DEPTH/2)
        glScalef(CELL_SIZE-2, CELL_SIZE-2, PIT_DEPTH)
        glutSolidCube(1)
        glPopMatrix()

        spike_spacing = (CELL_SIZE - 16) / 2.5
        for si in range(3):
            for sj in range(3):
                sx = wx - spike_spacing + si * spike_spacing
                sy = wy - spike_spacing + sj * spike_spacing
                glColor3f(0.25, 0.04, 0.04)
                glPushMatrix()
                glTranslatef(sx, sy, -PIT_DEPTH + SPIKE_BASE/2)
                glScalef(SPIKE_BASE, SPIKE_BASE, SPIKE_BASE)
                glutSolidCube(1)
                glPopMatrix()
                glColor3f(0.75, 0.08, 0.08)
                glPushMatrix()
                glTranslatef(sx, sy, -PIT_DEPTH + SPIKE_BASE + SPIKE_HEIGHT/2)
                glScalef(SPIKE_BASE*0.45, SPIKE_BASE*0.45, SPIKE_HEIGHT)
                glutSolidCube(1)
                glPopMatrix()
                tip_glow = 0.7 + 0.3 * math.sin(t*3.0 + si*1.3 + sj*0.9)
                glColor3f(1.0, tip_glow*0.15, 0.0)
                glPushMatrix()
                glTranslatef(sx, sy, -PIT_DEPTH + SPIKE_BASE + SPIKE_HEIGHT)
                gluSphere(gluNewQuadric(), SPIKE_BASE*0.38, 5, 5)
                glPopMatrix()

        glColor3f(0.06, 0.01, 0.01)
        glPushMatrix()
        glTranslatef(wx, wy, 0.8)
        glScalef(CELL_SIZE-6, CELL_SIZE-6, 1.5)
        glutSolidCube(1)
        glPopMatrix()

        glColor3f(pulse, pulse*0.12, 0.0)
        glPushMatrix()
        glTranslatef(wx, wy, 2.5)
        glScalef(CELL_SIZE+2, CELL_SIZE+2, 5)
        glBegin(GL_LINES)
        hx = (CELL_SIZE+2)*0.5; hy = (CELL_SIZE+2)*0.5; hz = 5*0.5
        corners = [
            (-hx,-hy,-hz),( hx,-hy,-hz),( hx, hy,-hz),(-hx, hy,-hz),
            (-hx,-hy, hz),( hx,-hy, hz),( hx, hy, hz),(-hx, hy, hz),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in edges:
            glVertex3fv(corners[a]); glVertex3fv(corners[b])
        glEnd()
        glPopMatrix()

        for cx2 in (-half+4, half-4):
            for cy2 in (-half+4, half-4):
                glColor3f(pulse*0.9, 0.0, 0.0)
                glPushMatrix()
                glTranslatef(wx+cx2, wy+cy2, 5)
                gluSphere(gluNewQuadric(), 3.5, 6, 6)
                glPopMatrix()

# member 2 
def draw_fall_scene():
    if not falling:
        return
    t = time.time()
    elapsed = time.time() - fall_start_time
    fraction = min(elapsed / RESPAWN_WAIT, 1.0)   # 0 → 1
    vig = fraction * 0.82
    draw_rect_2d(0,            0,            WINDOW_W,          int(WINDOW_H*vig*0.4), (0,0,0))
    draw_rect_2d(0,            WINDOW_H - int(WINDOW_H*vig*0.4), WINDOW_W, WINDOW_H,  (0,0,0))
    draw_rect_2d(0,            0,            int(WINDOW_W*vig*0.35), WINDOW_H,         (0,0,0))
    draw_rect_2d(WINDOW_W - int(WINDOW_W*vig*0.35), 0, WINDOW_W, WINDOW_H,            (0,0,0))

# member 3
def draw_player():
    pz       = fall_z if falling else player_pos[2]
    now      = time.time()
    flash_on = (now < hit_invincible_until) and (int(now*10) % 2 == 0)

    if not first_person:
        glPushMatrix()
        glTranslatef(player_pos[0], player_pos[1], pz)
        glRotatef(player_angle, 0, 0, 1)

        br, bg, bb = (1.0, 0.15, 0.15) if flash_on else (0.25, 0.50, 0.90)

        # Torso
        glPushMatrix(); glColor3f(br, bg, bb); glScalef(14,10,22); glutSolidCube(1); glPopMatrix()
        # Belt
        glPushMatrix(); glTranslatef(0,0,-6); glColor3f(0.15,0.12,0.05); glScalef(15,11,5); glutSolidCube(1); glPopMatrix()
        # Head
        glPushMatrix()
        glTranslatef(0, 0, 18)
        glColor3f(1.0, 0.82, 0.65)
        gluSphere(gluNewQuadric(), 7, 14, 14)
        # Eyes
        for ex in (-3.0, 3.0):
            glPushMatrix(); glTranslatef(ex, 6.8, 2); glColor3f(0.05,0.05,0.35); gluSphere(gluNewQuadric(), 1.8, 8, 8); glPopMatrix()
            glPushMatrix(); glTranslatef(ex, 7.5, 2); glColor3f(1.0,1.0,1.0); gluSphere(gluNewQuadric(), 0.7, 5, 5); glPopMatrix()
        glPopMatrix()
        # Arms
        for side in (-1, 1):
            glPushMatrix(); glTranslatef(side*10, 0, 3); glColor3f(br,bg,bb); glScalef(4,4,16); glutSolidCube(1); glPopMatrix()
        # Hands
        for side in (-1, 1):
            glPushMatrix(); glTranslatef(side*10, 0, -6); glColor3f(1.0,0.82,0.65); gluSphere(gluNewQuadric(), 3,8,8); glPopMatrix()
        # Legs
        for side in (-1, 1):
            glPushMatrix(); glTranslatef(side*4, 0, -16); glColor3f(0.12,0.18,0.55); glScalef(5,5,14); glutSolidCube(1); glPopMatrix()
        # Boots
        for side in (-1, 1):
            glPushMatrix(); glTranslatef(side*4, 1, -24); glColor3f(0.15,0.08,0.03); glScalef(6,8,5); glutSolidCube(1); glPopMatrix()

        glPopMatrix()
    else:
        # Gun barrel (first-person)
        glPushMatrix()
        glTranslatef(12, -9, -6); glRotatef(-8, 1,0,0)
        glColor3f(0.20, 0.20, 0.20); glScalef(5,4,22); glutSolidCube(1)
        glPopMatrix()
        # Gun handle
        glPushMatrix()
        glTranslatef(12, -9, -14); glRotatef(20, 1,0,0)
        glColor3f(0.30, 0.18, 0.05); glScalef(4,8,5); glutSolidCube(1)
        glPopMatrix()
        
# member 3
def draw_octahedron():
    v = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces = [(4,0,2),(4,2,1),(4,1,3),(4,3,0),(5,2,0),(5,1,2),(5,3,1),(5,0,3)]
    glBegin(GL_TRIANGLES)
    for f in faces:
        for vi in f:
            glVertex3fv(v[vi])
    glEnd()

# member 3
def draw_diamond():
    if diamond_found: return
    now  = time.time()
    spin = (now * 85.0) % 360.0
    bob  = math.sin(now * 2.2) * 7.0
    dx, dy = diamond_pos[0], diamond_pos[1]
    dz = 40.0 + bob

    # Pedestal
    glColor3f(0.30, 0.28, 0.35)
    glPushMatrix(); glTranslatef(dx, dy, 8); glScalef(22,22,16); glutSolidCube(1); glPopMatrix()
    glColor3f(0.55, 0.52, 0.60)
    glPushMatrix(); glTranslatef(dx, dy, 17); glScalef(14,14,4); glutSolidCube(1); glPopMatrix()

    # Orbiting glow ring
    ring_g = 0.55 + 0.35 * math.sin(now * 3.2)
    for i in range(16):
        ang = math.radians(i*22.5 + now*60)
        r_col = 0.3 + 0.6*(i%2)
        glColor3f(r_col*0.2, ring_g*r_col, 1.0)
        glPushMatrix()
        glTranslatef(dx + 30*math.cos(ang), dy + 30*math.sin(ang), dz - 5)
        gluSphere(gluNewQuadric(), 2.5, 5, 5)
        glPopMatrix()

    # Outer diamond 
    glPushMatrix()
    glTranslatef(dx, dy, dz)
    glRotatef(spin, 0, 0, 1); glRotatef(22, 1, 0, 0)
    glColor3f(0.0, 0.88, 1.0)
    glScalef(20, 20, 32)
    draw_octahedron()
    glPopMatrix()
    # Inner core 
    glPushMatrix()
    glTranslatef(dx, dy, dz)
    glRotatef(-spin*1.6, 0, 0, 1)
    glColor3f(0.90, 1.0, 1.0)
    glScalef(10, 10, 16)
    draw_octahedron()
    glPopMatrix()
    # Beacon sphere above
    beacon = 0.7 + 0.3 * math.sin(now * 4.0)
    glColor3f(1.0, beacon, 0.0)
    glPushMatrix(); glTranslatef(dx, dy, dz+65); gluSphere(gluNewQuadric(), 7, 12, 12); glPopMatrix()
    # Beacon pole
    glColor3f(0.6, 0.5, 0.0)
    glPushMatrix(); glTranslatef(dx, dy, dz+35); glScalef(2,2,55); glutSolidCube(1); glPopMatrix()

# member 1
def draw_enemies():
    now = time.time()
    for e in enemies:
        if not e['active']: continue
        e['scale'] += e['scale_dir']
        if e['scale'] > 1.20 or e['scale'] < 0.85:
            e['scale_dir'] = -e['scale_dir']

        ex, ey, ez = e['pos']
        boosted = now < _enemy_boost_until

        glPushMatrix()
        glTranslatef(ex, ey, ez)
        s = e['scale']
        glScalef(s, s, s)
        # Body — red (brighter when boosted)
        if boosted:
            glColor3f(1.0, 0.15, 0.0)
        else:
            glColor3f(0.85, 0.08, 0.08)
        gluSphere(gluNewQuadric(), 13, 20, 20)
        # Eyes — glowing yellow
        eye_glow = 0.7 + 0.3*math.sin(now*5 + ex)
        for sign in (-1, 1):
            glPushMatrix(); glTranslatef(sign*5, 11, 7)
            glColor3f(1.0, eye_glow, 0.0)
            gluSphere(gluNewQuadric(), 3.0, 8, 8)
            glPopMatrix()
            # Pupil
            glPushMatrix(); glTranslatef(sign*5, 13, 7)
            glColor3f(0.0, 0.0, 0.0)
            gluSphere(gluNewQuadric(), 1.2, 5, 5)
            glPopMatrix()
        # Horns (pointy scaled spheres)
        for sign in (-1, 1):
            glPushMatrix(); glTranslatef(sign*5, 0, 18)
            glColor3f(0.45, 0.0, 0.0)
            glScalef(1.0, 1.0, 3.5)
            gluSphere(gluNewQuadric(), 2.8, 6, 6)
            glPopMatrix()

        # Shadow disk on floor
        glPushMatrix()
        glTranslatef(0, 0, -ez + 1)
        glScalef(s*1.3, s*1.3, 0.3)
        glColor3f(0.0, 0.0, 0.0)
        gluSphere(gluNewQuadric(), 10, 10, 4)
        glPopMatrix()

        glPopMatrix()

# member 2 
def draw_bullets():
    now = time.time()
    for b in bullets:
        t_alive = now - b['time']
        pulse   = 0.7 + 0.3 * math.sin(t_alive * 30)
        glColor3f(0.10, pulse*0.85, 1.0)
        glPushMatrix()
        glTranslatef(*b['pos'])
        gluSphere(gluNewQuadric(), BULLET_RADIUS, 8, 8)
        glPopMatrix()
        # Trailing glow dot
        trail_x = b['pos'][0] - b['dir'][0]*12
        trail_y = b['pos'][1] - b['dir'][1]*12
        trail_z = b['pos'][2] - b['dir'][2]*12
        glColor3f(0.0, 0.3, 0.8)
        glPushMatrix()
        glTranslatef(trail_x, trail_y, trail_z)
        gluSphere(gluNewQuadric(), BULLET_RADIUS*0.5, 5, 5)
        glPopMatrix()

# member 2
def draw_big_ball():
    if not big_ball_active: return
    t  = time.time()
    gv = 0.25 + 0.20 * math.sin(t * 6)
    bx, by, bz = big_ball_pos
    q = gluNewQuadric()

    glPushMatrix(); glTranslatef(bx, by, bz)
    glColor3f(0.95, 0.35, 0.0)
    gluSphere(q, BIG_BALL_RADIUS + 9, 14, 14)
    glPopMatrix()

    # Main ball
    glPushMatrix(); glTranslatef(bx, by, bz)
    glColor3f(1.0, gv + 0.2, 0.0)
    gluSphere(q, BIG_BALL_RADIUS, 26, 26)
    glPopMatrix()
    # Hot inner core
    glPushMatrix(); glTranslatef(bx, by, bz)
    glColor3f(1.0, 1.0, gv*0.5)
    gluSphere(q, BIG_BALL_RADIUS * 0.45, 12, 12)
    glPopMatrix()
    # Shadow
    glPushMatrix()
    glTranslatef(bx, by, 1)
    glScalef(1.0, 1.0, 0.15)
    glColor3f(0.0, 0.0, 0.0)
    gluSphere(q, BIG_BALL_RADIUS * 1.1, 14, 8)
    glPopMatrix()

# member 3
def draw_map_markers():
    if not show_map_view: return
    markers = [
        (player_start_pos[0], player_start_pos[1], (0.1, 1.0, 0.1), "START"),
        (player_pos[0],       player_pos[1],       (0.2, 0.5, 1.0), "YOU"),
    ]
    if not diamond_found:
        markers.append((diamond_pos[0], diamond_pos[1], (1.0, 1.0, 0.0), "GEM"))
    for mx, my, col, _ in markers:
        glPushMatrix(); glTranslatef(mx, my, 200)
        glColor3f(*col)
        gluSphere(gluNewQuadric(), 22, 14, 14)
        glPopMatrix()
        # Vertical pole
        glPushMatrix(); glTranslatef(mx, my, 110)
        glColor3f(*col)
        glScalef(3, 3, 180)
        glutSolidCube(1)
        glPopMatrix()
        
# member 3
def draw_hud():
    # Top bar background 
    draw_rect_2d(0, WINDOW_H-50, WINDOW_W, WINDOW_H, (0.05, 0.05, 0.08))
    # Accent line under bar
    draw_rect_2d(0, WINDOW_H-52, WINDOW_W, WINDOW_H-50, (0.3, 0.6, 1.0))

    #  green → yellow → red
    if game_time > 90:
        t_col = (0.3, 1.0, 0.3)
    elif game_time > 30:
        t_col = (1.0, 0.85, 0.1)
    else:
        t_col = (1.0, 0.2, 0.1)
    draw_text_2d(10,  WINDOW_H-32, f"TIME: {game_time}s",    color=t_col)
    draw_text_2d(145, WINDOW_H-32, f"LIVES: {lives}",        color=(1.0, 0.35, 0.35))
    draw_text_2d(280, WINDOW_H-32,
                 f"ENEMIES: {sum(1 for e in enemies if e['active'])}",
                 color=(1.0, 0.65, 0.2))
    draw_text_2d(440, WINDOW_H-32,
                 f"QUIZ: {MAX_QUESTIONS - questions_answered} left",
                 color=(0.4, 1.0, 0.5))
    if big_ball_active:
        draw_text_2d(620, WINDOW_H-32, "BALL: ACTIVE!", color=(1.0, 0.4, 0.0))
    else:
        wait = max(0.0, BIG_BALL_RESPAWN_DELAY - (time.time()-big_ball_respawn_timer))
        draw_text_2d(620, WINDOW_H-32, f"BALL: {wait:.1f}s", color=(0.5, 0.5, 0.5))
    draw_text_2d(820, WINDOW_H-32, f"SCORE: {score}", color=(1.0, 1.0, 0.4))

    # Status messages below bar 
    y = WINDOW_H - 78
    if cheat_mode:
        draw_text_2d(10, y, "[ CHEAT MODE ON ]", color=(0.0, 1.0, 1.0)); y -= 24
    if show_map_view:
        rem = max(0.0, MAP_VIEW_DURATION - (time.time()-map_view_start))
        draw_text_2d(10, y, f"[ MAP VIEW: {rem:.0f}s ]", color=(1.0, 1.0, 0.2)); y -= 24
    if time.time() < hit_invincible_until:
        draw_text_2d(440, y, "** HIT! **", GLUT_BITMAP_HELVETICA_18, (1.0, 0.1, 0.0)); y -= 24
    if time.time() < _enemy_boost_until:
        draw_text_2d(10, y, "[ ENEMIES BOOSTED! ]", color=(1.0, 0.2, 0.2)); y -= 24

    if falling:
        elapsed = time.time() - fall_start_time
        left    = max(0.0, RESPAWN_WAIT - elapsed)
        draw_text_2d(290, WINDOW_H//2 + 60,
                     "** FALLING INTO SPIKE PIT! **",
                     GLUT_BITMAP_HELVETICA_18, (1.0, 0.15, 0.0))
        draw_text_2d(360, WINDOW_H//2 + 20,
                     f"Respawning in {left:.1f}s ...",
                     GLUT_BITMAP_HELVETICA_18, (1.0, 0.70, 0.0))

    if first_person and not game_over:
        cx, cy = WINDOW_W//2, WINDOW_H//2
        draw_rect_2d(cx-16, cy-2, cx+16, cy+2, (1,1,1))
        draw_rect_2d(cx-2, cy-16, cx+2, cy+16, (1,1,1))
        draw_rect_2d(cx-6,  cy-2, cx+6,  cy+2, (0.3,0.8,1.0))

    if not game_over:
        draw_rect_2d(0, 0, WINDOW_W, 24, (0.03, 0.03, 0.06))
        if first_person:
            hint = "W/S:Move  A/D:Turn  Arrows:Look  SPACE/LMB:Shoot  RMB:Quiz  C:Cheat  R:Reset "
        else:
            hint = "W/S:Move  A/D:Turn  Arrows:Camera  RMB:Quiz  C:Cheat  R:Reset "
        draw_text_2d(6, 6, hint, color=(0.45, 0.45, 0.55))

    if game_over:
        draw_rect_2d(100, 300, 900, 510, (0.03, 0.03, 0.10))
        draw_rect_2d(100, 490, 900, 510, (0.2,  0.4,  1.0 ))
        draw_rect_2d(100, 300, 900, 320, (0.2,  0.4,  1.0 ))
        if diamond_found:
            msg = "** YOU FOUND THE DIAMOND  -  YOU WIN! **"
            col = (0.2, 1.0, 0.4)
        elif lives <= 0:
            msg = "** NO LIVES LEFT  -  GAME OVER **"
            col = (1.0, 0.2, 0.2)
        else:
            msg = "** TIME'S UP  -  GAME OVER **"
            col = (1.0, 0.75, 0.0)
        draw_text_2d(165, 430, msg, GLUT_BITMAP_HELVETICA_18, col)
        draw_text_2d(280, 360,
                     "Press  R  to restart",
                     color=(0.85, 0.85, 0.90))

# member 3
def draw_question_overlay():
    if not question_active or not current_question: return
    draw_rect_2d(130, 240, 870, 560, (0.06, 0.06, 0.28))
    draw_rect_2d(130, 525, 870, 560, (0.12, 0.12, 0.48))
    draw_rect_2d(130, 240, 870, 270, (0.12, 0.12, 0.48))
    draw_text_2d(340, 533, "-- QUIZ CHALLENGE --",
                 GLUT_BITMAP_HELVETICA_18, (1.0, 1.0, 0.3))
    draw_text_2d(170, 480, current_question["q"],
                 GLUT_BITMAP_HELVETICA_18, (1.0, 1.0, 1.0))
    cols = [(0.3, 1.0, 0.4), (0.3, 0.7, 1.0)]
    for i, opt in enumerate(current_question["opts"]):
        draw_text_2d(220, 400 - i*70,
                     f"{i+1}.  {opt}",
                     GLUT_BITMAP_HELVETICA_18, cols[i])
    draw_text_2d(300, 255,
                 "Press  1  or  2  to answer",
                 GLUT_BITMAP_HELVETICA_18, (0.70, 0.70, 0.75))

# member 2
def check_hole_trap():
    global falling, fall_z, fall_start_time, fall_px, fall_py
    if falling or cheat_mode: return
    pr, pc = world_to_cell(player_pos[0], player_pos[1])
    if (pr, pc) in hole_cells:
        falling         = True
        fall_z          = player_pos[2]
        fall_start_time = time.time()
        fall_px, fall_py = cell_to_world(pr, pc)

# member 2 
def update_falling():
    global falling, fall_z, player_pos, lives, game_over, hit_invincible_until
    if not falling: return

    fall_z -= FALL_SPEED * dt           # smooth downward motion each frame

    if time.time() - fall_start_time >= RESPAWN_WAIT:
        falling              = False
        fall_z               = 10.0
        player_pos[0]        = player_start_pos[0]
        player_pos[1]        = player_start_pos[1]
        player_pos[2]        = 10.0
        lives               -= 1
        hit_invincible_until = time.time() + HIT_INVINCIBLE_SECS
        push_enemies_from_spawn()
        if lives <= 0:
            game_over = True

# member 3
def check_diamond_pickup():
    global diamond_found, game_over, score
    if diamond_found: return
    if math.hypot(player_pos[0]-diamond_pos[0], player_pos[1]-diamond_pos[1]) < 34:
        diamond_found = True
        score        += 1000
        game_over     = True

# member 1 
def check_enemy_touch():
    global lives, game_over, hit_invincible_until
    if time.time() < hit_invincible_until: return
    px, py = player_pos[0], player_pos[1]
    for e in enemies:
        if not e['active']: continue
        dist = math.hypot(px-e['pos'][0], py-e['pos'][1])
        if dist < 24:
            lives               -= 1
            hit_invincible_until = time.time() + HIT_INVINCIBLE_SECS
            ang = math.atan2(e['pos'][1]-py, e['pos'][0]-px) if dist>1 else random.uniform(0,2*math.pi)
            e['pos'][0] = px + 95*math.cos(ang)
            e['pos'][1] = py + 95*math.sin(ang)
            e['path'] = []
            if lives <= 0: game_over = True
            break

# member 2 
def update_bullets():
    now = time.time()
    for b in bullets:
        el = now - b['time']
        b['pos'] = [b['start'][0]+b['dir'][0]*BULLET_SPEED*el,
                    b['start'][1]+b['dir'][1]*BULLET_SPEED*el,
                    b['start'][2]+b['dir'][2]*BULLET_SPEED*el]

# member 2 
def check_bullet_hits():
    global score
    for b in bullets[:]:
        bx, by, bz = b['pos']
        if touches_wall(bx, by, probe=4):
            if b in bullets: bullets.remove(b)
            continue
        for e in enemies:
            if not e['active']: continue
            if math.sqrt((bx-e['pos'][0])**2+(by-e['pos'][1])**2+(bz-e['pos'][2])**2) < 22:
                e['active'] = False
                score      += 100
                if b in bullets: bullets.remove(b)
                break

# member 2
def update_bullets_lifetime():
    now = time.time()
    for b in bullets[:]:
        if (now-b['time'])*BULLET_SPEED > BULLET_RANGE:
            if b in bullets: bullets.remove(b)

# member 1 
def update_enemy_ai():
    now    = time.time()
    gr, gc = world_to_cell(player_pos[0], player_pos[1])
    spd    = ENEMY_BASE_SPEED * 1.65 if now < _enemy_boost_until else ENEMY_BASE_SPEED
    for e in enemies:
        if not e['active']: continue
        e['speed'] = spd
        er, ec     = world_to_cell(e['pos'][0], e['pos'][1])
        if (not e['path']) or (now-e['last_repath'] > REPATH_INTERVAL):
            e['path']        = astar_path((er,ec), (gr,gc))
            e['last_repath'] = now
        if not e['path']: continue
        wr, wc = e['path'][0]
        wx, wy = cell_to_world(wr, wc)
        dx_e  = wx - e['pos'][0]
        dy_e  = wy - e['pos'][1]
        dist  = math.hypot(dx_e, dy_e)
        step  = spd * dt
        if dist < step+2:
            e['pos'][0], e['pos'][1] = wx, wy
            e['path'].pop(0)
        else:
            e['pos'][0] += (dx_e/dist)*step
            e['pos'][1] += (dy_e/dist)*step
# member 2
def update_big_ball():
    global big_ball_active, big_ball_spawned, big_ball_pos, big_ball_respawn_timer
    global lives, game_over, hit_invincible_until
    now = time.time()
    if not big_ball_spawned:
        if now - game_start_time >= BIG_BALL_SPAWN_DELAY:
            big_ball_spawned = True
            big_ball_active  = True
            ang = random.uniform(0, 2*math.pi)
            big_ball_pos[:] = [player_pos[0]+BIG_BALL_MIN_SPAWN_DIST*math.cos(ang),
                                player_pos[1]+BIG_BALL_MIN_SPAWN_DIST*math.sin(ang), 50.0]
        return
    if not big_ball_active:
        if now - big_ball_respawn_timer >= BIG_BALL_RESPAWN_DELAY:
            big_ball_active = True
            ang  = random.uniform(0, 2*math.pi)
            dist = random.uniform(BIG_BALL_MIN_SPAWN_DIST, BIG_BALL_MIN_SPAWN_DIST+180)
            big_ball_pos[:] = [player_pos[0]+dist*math.cos(ang),
                                player_pos[1]+dist*math.sin(ang), 50.0]
        return
    px, py = player_pos[0], player_pos[1]
    bx, by = big_ball_pos[0], big_ball_pos[1]
    dist   = math.hypot(px-bx, py-by)
    if dist > 0.01:
        step = BIG_BALL_SPEED * dt
        big_ball_pos[0] += (px-bx)/dist*step
        big_ball_pos[1] += (py-by)/dist*step
    if dist < BIG_BALL_RADIUS+18 and now >= hit_invincible_until:
        lives                 -= 1
        big_ball_active        = False
        big_ball_respawn_timer = now
        hit_invincible_until   = now + HIT_INVINCIBLE_SECS
        if lives <= 0: game_over = True

# member 2
def update_holes():
    global hole_cells, _hole_phase_start, _holes_visible
    now     = time.time()
    elapsed = now - _hole_phase_start

    if _holes_visible:
        # Holes are showing , hide them after HOLE_VISIBLE_SECS
        if elapsed >= HOLE_VISIBLE_SECS:
            hole_cells        = []          
            _holes_visible    = False
            _hole_phase_start = now
    else:
        # Holes are hidden , show new random subset after HOLE_HIDDEN_SECS
        if elapsed >= HOLE_HIDDEN_SECS:
            pool = list(_all_hole_cells)
            random.shuffle(pool)
            hole_cells        = pool[:HOLE_COUNT]
            _holes_visible    = True
            _hole_phase_start = now


def update_timers():
    global game_time, game_over, first_person, show_map_view
    elapsed   = time.time() - game_start_time
    game_time = max(0, 180 - int(elapsed))
    if game_time == 0 and not game_over: game_over = True
    if not first_person and elapsed >= AUTO_FP_DELAY: first_person = True
    if show_map_view and (time.time()-map_view_start) > MAP_VIEW_DURATION:
        show_map_view = False
        first_person  = True   

#  QUIZ  (Member 3)
def trigger_question():
    global question_active, current_question, questions_answered
    if questions_answered >= MAX_QUESTIONS or not remaining_questions: return
    idx              = random.randrange(len(remaining_questions))
    current_question = remaining_questions.pop(idx)
    question_active  = True
    questions_answered += 1

# member 3
def resolve_answer(choice):
    global question_active, show_map_view, map_view_start, _enemy_boost_until, score
    global first_person
    if not question_active or not current_question: return
    question_active = False
    if choice == current_question["ans"]:
        show_map_view  = True
        map_view_start = time.time()
        first_person   = False   # switch to top-down map view automatically
        score         += 50
    else:
        _enemy_boost_until = time.time() + ENEMY_BOOST_DURATION

#  SHOOTING  (member 2)
def shoot():
    global last_shot_time
    now = time.time()
    if now - last_shot_time < SHOT_COOLDOWN: return
    last_shot_time = now
    rad   = math.radians(player_angle)
    pr    = math.radians(fp_pitch)
    horiz = math.cos(pr)
    direction = (math.cos(rad)*horiz, math.sin(rad)*horiz, math.sin(pr))
    start = (player_pos[0]+direction[0]*16,
             player_pos[1]+direction[1]*16,
             player_pos[2]+8)
    bullets.append({'start':start, 'pos':list(start), 'dir':direction, 'time':now})

#  CAMERA  (Member 3)
def setup_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = WINDOW_W / WINDOW_H

    if show_map_view:
        margin = HALF_GRID * 0.12
        glOrtho(-HALF_GRID-margin, HALF_GRID+margin,
                -HALF_GRID-margin, HALF_GRID+margin,
                -600, 2500)
    elif falling:
        gluPerspective(85, aspect, 1.0, 1500)
    elif first_person:
        gluPerspective(90,aspect, 0.5, 1500)
    else:
        gluPerspective(75, aspect, 0.5, 1500)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    if show_map_view:
        gluLookAt(0, 0, 1200,  0, 0, 0,  0, 1, 0)

    elif falling:
        elapsed  = time.time() - fall_start_time
        fraction = min(elapsed / RESPAWN_WAIT, 1.0)

        eye_z = 120.0 - fraction * (PIT_DEPTH * 0.80)
        look_z = -PIT_DEPTH + SPIKE_HEIGHT + 10
        gluLookAt(fall_px, fall_py, eye_z,
                  fall_px, fall_py, look_z,
                  0, 1, 0)

    elif first_person:
        pz    = player_pos[2]
        eye_z = pz + 8
        rad   = math.radians(player_angle)
        pr   = math.radians(fp_pitch)
        fx    = math.cos(rad) * math.cos(pr)
        fy    = math.sin(rad) * math.cos(pr)
        fz    = math.sin(pr)
        gluLookAt(player_pos[0], player_pos[1], eye_z,
                  player_pos[0]+fx, player_pos[1]+fy, eye_z+fz,
                  0, 0, 1)
    else:
        gluLookAt(camera_x, camera_y, camera_z,  0, 0, 0,  0, 0, 1)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glViewport(0, 0, WINDOW_W, WINDOW_H)

    setup_camera()
    draw_floor()
    draw_holes()           
    draw_maze()
    draw_entrance()
    draw_torches()
    draw_diamond()
    draw_player() 
    draw_enemies()
    draw_bullets()
    draw_big_ball()
    draw_map_markers()

    draw_fall_scene()    
    draw_hud()
    draw_question_overlay()

    glutSwapBuffers()


def idle():
    global _last_idle_time, dt
    now             = time.time()
    dt              = min(now - _last_idle_time, 0.05)
    _last_idle_time = now

    if not game_over:
        update_timers()
        update_holes()
        update_falling()
        if not falling:
            check_hole_trap()
        update_bullets()
        check_bullet_hits()
        update_bullets_lifetime()
        if not falling:
            update_enemy_ai()
            update_big_ball()
            check_enemy_touch()
            check_diamond_pickup()

    glutPostRedisplay()

# member 1
# member 2
def keyboardListener(key, x, y):
    global player_angle, cheat_mode, first_person, game_over

    if key in (b'r', b'R'):
        reset_game()
        build_torch_positions()
        glutPostRedisplay()
        return
    if game_over:
        return

    k = key.decode('utf-8').lower()

    if question_active:
        if k == '1': resolve_answer(0)
        elif k == '2': resolve_answer(1)
        glutPostRedisplay()
        return

    if falling:
        return

    if k in ('w','s'):
        sign = 1 if k=='w' else -1
        step = sign * player_speed * dt
        dx   = step * math.cos(math.radians(player_angle))
        dy   = step * math.sin(math.radians(player_angle))
        nx   = player_pos[0] + dx
        ny   = player_pos[1] + dy
        if not touches_wall(nx, ny) or cheat_mode:
            player_pos[0], player_pos[1] = nx, ny
            check_hole_trap()
            check_diamond_pickup()
    elif k=='a': player_angle = (player_angle + 4) % 360
    elif k=='d': player_angle = (player_angle - 4) % 360
    elif k=='c': cheat_mode = not cheat_mode
    elif k==' ':
        if first_person: shoot()

    glutPostRedisplay()

# member 2
def specialKeyListener(key, x, y):
    global camera_x, camera_y, camera_z, fp_pitch, player_angle
    if game_over: return
    if first_person:
        if   key== GLUT_KEY_UP:    fp_pitch     = min(fp_pitch+3, 70)
        elif key== GLUT_KEY_DOWN:  fp_pitch     = max(fp_pitch-3, -70)
        elif key== GLUT_KEY_LEFT:  player_angle = (player_angle+4) % 360
        elif key== GLUT_KEY_RIGHT: player_angle = (player_angle-4) % 360
    else:
        step = 25
        if   key== GLUT_KEY_UP:    camera_y += step
        elif key== GLUT_KEY_DOWN:  camera_y  = max(80, camera_y-step)
        elif key== GLUT_KEY_LEFT:  camera_x -= step
        elif key== GLUT_KEY_RIGHT: camera_x += step
    glutPostRedisplay()

# member 2
def mouseListener(button, state, x, y):
    if game_over: return
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        if first_person: shoot()
    elif button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
        if question_active: return
        if questions_answered < MAX_QUESTIONS and remaining_questions:
            trigger_question()
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_W, WINDOW_H)
    glutInitWindowPosition(100, 50)
    glutCreateWindow(b"Diamond Quest")

    glEnable(GL_DEPTH_TEST)

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboardListener)
    glutSpecialFunc(specialKeyListener)
    glutMouseFunc(mouseListener)

    reset_game()
    build_torch_positions()
    glutMainLoop()

if __name__ == "__main__":
    main()