/** LabDex stand-in for PokeStonks' `game-core`: just the creature roster the art engine needs. */
export interface Species { id: string; ticker: string; name: string; colors: [string, string]; type: string }

export const SPECIES_LIST: Species[] = [
  { id: 'reader',   ticker: 'READER',   name: 'Lorecat',     colors: ['#8b6cf0', '#ddd3ff'], type: 'psychic' },
  { id: 'planner',  ticker: 'PLANNER',  name: 'Plandrop',    colors: ['#4aa8ff', '#cfe8ff'], type: 'water' },
  { id: 'careful',  ticker: 'CAREFUL',  name: 'Cogbot',      colors: ['#9aa7ab', '#e3e8ea'], type: 'steel' },
  { id: 'literal',  ticker: 'LITERAL',  name: 'Runnix',      colors: ['#d9b27c', '#f1e0c5'], type: 'normal' },
  { id: 'hasty',    ticker: 'HASTY',    name: 'Hastelo',     colors: ['#ff5436', '#ffb93a'], type: 'fire' },
  { id: 'debugger', ticker: 'DEBUGGER', name: 'Fixbug',      colors: ['#6fbf3a', '#f2e9a0'], type: 'bug' },
  { id: 'analyst',  ticker: 'ANALYST',  name: 'Voltyst',     colors: ['#ffc21a', '#fff0b0'], type: 'electric' },
  { id: 'tuner',    ticker: 'TUNER',    name: 'Sweepsprout', colors: ['#169b48', '#b9f0cb'], type: 'grass' },
];
const byId = new Map(SPECIES_LIST.map((s) => [s.id, s]));
export const isSpeciesId = (id: string): boolean => byId.has(id);
export const getSpecies = (id: string): Species => byId.get(id) ?? SPECIES_LIST[0];
