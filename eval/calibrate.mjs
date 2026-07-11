import { calibrate } from './judge.mjs';
console.log('\n  CALIBRATING JUDGE  (✓ = known-good grid, should score high · ✗ = known-bad, should score low)\n');
const { fit } = await calibrate();
process.exit(fit ? 0 : 1);
