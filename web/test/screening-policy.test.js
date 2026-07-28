import assert from "node:assert/strict";
import test from "node:test";

import {
  LESION_PRESENCE_THRESHOLD,
  lesionPresenceEvidence,
  patternConcernScore,
  screeningDecision,
} from "../src/screening-policy.js";

function syntheticImage(centerColor, borderColor, size = 128) {
  const pixels = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const radius = Math.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2) / (size / 2);
      const color = radius <= 0.64 ? centerColor : borderColor;
      const offset = (y * size + x) * 4;
      pixels.set([...color, 255], offset);
    }
  }
  return pixels;
}

test("uniform skin does not pass the centered-lesion evidence gate", () => {
  const pixels = syntheticImage([165, 125, 105], [165, 125, 105]);
  assert.ok(lesionPresenceEvidence(pixels, 128, 128) < LESION_PRESENCE_THRESHOLD);
});

test("a clearly contrasting centered spot passes the evidence gate", () => {
  const pixels = syntheticImage([75, 45, 35], [175, 135, 115]);
  assert.ok(lesionPresenceEvidence(pixels, 128, 128) > LESION_PRESENCE_THRESHOLD);
});

test("the display scale no longer turns moderate ensemble evidence into 9 out of 10", () => {
  assert.equal(patternConcernScore(0.5), 4);
  assert.equal(patternConcernScore(0.7), 6);
  assert.equal(patternConcernScore(0.9), 9);
});

test("the melanoma safety head can flag review without inflating the displayed ensemble score", () => {
  const decision = screeningDecision(
    { higherConcern: 0.3, melanoma: 0.01 },
    { higher_concern: 0.4477, melanoma: 0.0045 },
  );
  assert.equal(decision.reviewRecommended, true);
  assert.equal(decision.higherConcernFlag, false);
  assert.equal(decision.melanomaSafetyFlag, true);
  assert.equal(patternConcernScore(0.3), 2);
});
