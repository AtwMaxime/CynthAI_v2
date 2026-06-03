/**
 * generate_team_pool.js
 *
 * Generates a pool of official Gen 9 Random Battle teams using the
 * pokemon-showdown team generator, and writes them to data/team_pool.json.
 *
 * Usage:
 *   node scripts/generate_team_pool.js [--n 5000] [--out data/team_pool.json]
 *
 * Run from CynthAI_v2/ root.
 */

const path = require('path');
const fs   = require('fs');

// Resolve pokemon-showdown relative to this script's location (CynthAI_v2/scripts/)
const PS_DIR = path.resolve(__dirname, '..', '..', 'pokemon-showdown');

const { Teams, PRNG } = require(path.join(PS_DIR, 'dist', 'sim', 'index'));

// ── CLI args ──────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
function getArg(flag, defaultVal) {
    const idx = args.indexOf(flag);
    return idx !== -1 ? args[idx + 1] : defaultVal;
}

const N        = parseInt(getArg('--n',   '5000'), 10);
const OUT_PATH = path.resolve(__dirname, '..', getArg('--out', 'data/team_pool.json'));
const FORMAT   = getArg('--format', 'gen9randombattle');

// ── Generate ──────────────────────────────────────────────────────────────────
console.log(`Generating ${N} teams for ${FORMAT}...`);

const packed_teams = [];

for (let seed = 0; seed < N; seed++) {
    const prng = new PRNG([0, 0, seed >> 16, seed & 0xffff]);
    try {
        const team   = Teams.getGenerator(FORMAT, prng).getTeam();
        const packed = Teams.pack(team);
        if (packed && packed.length > 0) {
            packed_teams.push(packed);
        }
    } catch (e) {
        // skip invalid seeds silently
    }

    if ((seed + 1) % 500 === 0) {
        process.stdout.write(`  ${seed + 1}/${N}\r`);
    }
}

console.log(`\nGenerated ${packed_teams.length} teams.`);

// ── Write ─────────────────────────────────────────────────────────────────────
fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
fs.writeFileSync(OUT_PATH, JSON.stringify(packed_teams, null, 0));
console.log(`Saved to ${OUT_PATH}  (${(fs.statSync(OUT_PATH).size / 1024).toFixed(1)} KB)`);
