export const LESION_PRESENCE_THRESHOLD = 0.12;

export function lesionPresenceEvidence(pixels, width, height) {
  const centerSum = [0, 0, 0];
  const borderSum = [0, 0, 0];
  const globalSum = [0, 0, 0];
  const globalSquaredSum = [0, 0, 0];
  let centerCount = 0;
  let borderCount = 0;
  let globalCount = 0;
  let centerLuminance = 0;
  let borderLuminance = 0;
  let luminanceSum = 0;
  let luminanceSquaredSum = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const rgb = [pixels[offset], pixels[offset + 1], pixels[offset + 2]];
      const normalizedX = (x + 0.5 - width / 2) / (width / 2);
      const normalizedY = (y + 0.5 - height / 2) / (height / 2);
      const radius = Math.sqrt(normalizedX * normalizedX + normalizedY * normalizedY);
      const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];

      for (let channel = 0; channel < 3; channel += 1) {
        globalSum[channel] += rgb[channel];
        globalSquaredSum[channel] += rgb[channel] * rgb[channel];
      }
      luminanceSum += luminance;
      luminanceSquaredSum += luminance * luminance;
      globalCount += 1;

      if (radius <= 0.64) {
        for (let channel = 0; channel < 3; channel += 1) centerSum[channel] += rgb[channel];
        centerLuminance += luminance;
        centerCount += 1;
      } else if (radius >= 1.36) {
        for (let channel = 0; channel < 3; channel += 1) borderSum[channel] += rgb[channel];
        borderLuminance += luminance;
        borderCount += 1;
      }
    }
  }

  if (!centerCount || !borderCount || !globalCount) return 0;
  const centerMean = centerSum.map((value) => value / centerCount);
  const borderMean = borderSum.map((value) => value / borderCount);
  const standardizedColorDifference = Math.sqrt(
    centerMean.reduce((total, value, channel) => {
      const mean = globalSum[channel] / globalCount;
      const variance = Math.max(0, globalSquaredSum[channel] / globalCount - mean * mean);
      const stabilizedDeviation = Math.sqrt(variance) + 8;
      return total + ((value - borderMean[channel]) / stabilizedDeviation) ** 2;
    }, 0) / 3,
  );
  const luminanceMean = luminanceSum / globalCount;
  const luminanceDeviation =
    Math.sqrt(Math.max(0, luminanceSquaredSum / globalCount - luminanceMean * luminanceMean)) + 5;
  const standardizedLuminanceDifference =
    Math.abs(centerLuminance / centerCount - borderLuminance / borderCount) / luminanceDeviation;
  const rawColorDifference =
    Math.sqrt(centerMean.reduce((total, value, channel) => total + (value - borderMean[channel]) ** 2, 0)) /
    40;

  return Math.max(standardizedColorDifference, standardizedLuminanceDifference, rawColorDifference);
}

export function patternConcernScore(higherConcernScore) {
  const boundedEvidence = Math.max(0, Math.min(1, higherConcernScore));
  return Math.max(1, Math.min(10, Math.round(1 + 9 * boundedEvidence ** 1.7)));
}

export function screeningDecision(scores, thresholds) {
  return {
    reviewRecommended:
      scores.higherConcern >= thresholds.higher_concern || scores.melanoma >= thresholds.melanoma,
    higherConcernFlag: scores.higherConcern >= thresholds.higher_concern,
    melanomaSafetyFlag: scores.melanoma >= thresholds.melanoma,
  };
}
