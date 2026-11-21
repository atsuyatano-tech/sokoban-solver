// Sokoban solver: push-optimal, room by room. Every level here is a one-wide hallway with 1-4 rooms
// off it, each room holding as many boxes as goals and opening through one door. A box pushed through
// a door into the hallway can never return, so each box stays in its room and the level's push optimum
// P* is the sum of its rooms' optima. The hallway squares are exactly the dead squares (a box there
// reaches no goal) of the player's row: cutting them leaves one component per room, door included.
// Each room is solved by breadth-first search over pushes (state: box squares + lowest square of the
// player's reachable region), which meets states in push order, so the first solved state is optimal.
// Prunes, both sound: never push a box onto a dead square; drop pushes freezing a 2x2 block of walls
// and boxes with a box off goal. All budgets are fixed state counts -- no clock, no randomness -- so
// two runs agree byte for byte. Fallbacks: a whole-level push search, else an empty (valid) answer.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
/** [dx, dy, walk letter, push letter]; the fixed order every tie is broken by. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
const ROOM_STATES = 900_000; // stored-state budget per room search
const LEVEL_STATES = 1_500_000; // stored-state budget for the whole-level fallback

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** XSB text to a level: walls (outer frame always wall), goals, boxes, player. */
function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals). */
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let i = 0; i < queue.length; i++) {
    const x = queue[i];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Flood the player's region from `from` around walls and boxes; stamp `seen`, return its lowest square. */
function reach(w: number, wall: Uint8Array, occ: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const q of [p - w, p + w, p - 1, p + 1])
      if (!wall[q] && !occ[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 of walls and boxes, a box among them off goal. */
function frozen(w: number, wall: Uint8Array, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower LURD; null when unreachable. */
function walkTo(w: number, wall: Uint8Array, occ: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let i = 0; i < queue.length; i++) {
    const p = queue[i];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Push-optimal solution of one (sub-)level: {pushes, moves} (upper letter per push), or null when the
 * search stores `cap` states without solving it. BFS over pushes: the first solved state is optimal.
 */
function solvePush(w: number, wall: Uint8Array, goals: number[], startBoxes: number[], startPlayer: number, cap: number): { pushes: number; moves: string } | null {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  if (startBoxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b] === 1);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };
  const nb = startBoxes.length;
  // Numeric state keys while the packed boxes stay exact in a double; string keys past that.
  const numeric = size <= 256 && nb <= 5;
  const span = 256 ** nb;
  const pack = (bs: number[]): number => {
    let k = 0;
    for (const b of bs) k = k * 256 + b;
    return k;
  };
  const unpack = (k: number): number[] => {
    const bs = new Array<number>(nb);
    for (let i = nb - 1; i >= 0; i--) {
      bs[i] = k % 256;
      k = Math.floor(k / 256);
    }
    return bs;
  };
  const keyOf = (bs: number[], low: number): number | string => (numeric ? low * span + pack(bs) : low + '|' + bs.join(','));
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const sPack: number[] = [numeric ? pack(startBoxes) : 0];
  const sBoxes: number[][] = [startBoxes.slice()];
  const sPlayer: number[] = [startPlayer];
  const sParent: number[] = [-1];
  const boxesAt = (s: number): number[] => (numeric ? unpack(sPack[s]) : sBoxes[s]);
  const index = new Map<number | string, number>();
  for (const b of startBoxes) occ[b] = 1;
  index.set(keyOf(startBoxes, reach(w, wall, occ, startPlayer, seen, ++stamp, stack)), 0);
  for (const b of startBoxes) occ[b] = 0;
  let found = -1;
  for (let head = 0; head < sPlayer.length && found < 0; head++) {
    const boxes = boxesAt(head);
    for (const b of boxes) occ[b] = 1;
    reach(w, wall, occ, sPlayer[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const froz = frozen(w, wall, occ, goal, t);
        let key: number | string | null = null;
        let next: number[] | null = null;
        if (!froz) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(w, wall, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0;
        occ[b] = 1;
        if (froz || key === null || index.has(key)) continue;
        if (sPlayer.length >= cap) {
          for (const x of boxes) occ[x] = 0;
          return null;
        }
        index.set(key, sPlayer.length);
        if (numeric) sPack.push(pack(next));
        else sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(head);
        if (solved(next)) found = sPlayer.length - 1;
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = boxesAt(s);
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occ[b] = 1;
    const walk = walkTo(w, wall, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway) out of the floor; each
 * component left holding a box is a room, door square included. Null when the level is no set of
 * rooms: a box in the hallway, a goal cut off, or a component whose box and goal counts differ.
 */
function roomsOf(L: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = L;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++)
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** The whole-level fallback: one push search over the level as it stands. */
function solveWhole(L: Level): string {
  const whole = solvePush(L.w, L.wall, L.goals, L.boxes, L.player, LEVEL_STATES);
  return whole === null ? '' : whole.moves;
}

/** One level's moves: room by room, the player walking the hallway between doors. */
function solve(text: string): string {
  const L = parse(text);
  const rooms = roomsOf(L);
  if (rooms === null || rooms.length === 0) return solveWhole(L);
  const hallRow = Math.floor(L.player / L.w);
  let boxesNow = L.boxes.slice();
  let player = L.player;
  let moves = '';
  const occ = new Uint8Array(L.wall.length);
  for (const room of rooms) {
    // The room's door opens onto one hallway square; walk there around the boxes as they stand.
    let hallSq = -1;
    for (const s of room.squares) {
      for (const [dx, dy] of DIRS) {
        const q = s + dy * L.w + dx;
        if (Math.floor(q / L.w) === hallRow && !L.wall[q]) {
          hallSq = q;
          break;
        }
      }
      if (hallSq >= 0) break;
    }
    if (hallSq < 0) return solveWhole(L);
    occ.fill(0);
    for (const b of boxesNow) occ[b] = 1;
    const walk = walkTo(L.w, L.wall, occ, player, hallSq);
    if (walk === null) return solveWhole(L);
    moves += walk;
    player = hallSq;
    // The room's own search: only its squares and the hallway are walkable, the rest is wall. Its
    // boxes can never leave through the door -- the hallway squares are dead for its goals.
    const wall2 = L.wall.slice();
    const mine = new Set(room.squares);
    for (let i = 0; i < wall2.length; i++) {
      if (wall2[i] || mine.has(i) || Math.floor(i / L.w) === hallRow) continue;
      wall2[i] = 1;
    }
    const sub = solvePush(L.w, wall2, room.goals, room.boxes.slice().sort((a, b) => a - b), player, ROOM_STATES);
    if (sub === null) return solveWhole(L);
    moves += sub.moves;
    player = endOf(L.w, player, sub.moves);
    const done = new Set(room.boxes);
    boxesNow = boxesNow.filter((b) => !done.has(b)).concat(room.goals);
    boxesNow.sort((a, b) => a - b);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
