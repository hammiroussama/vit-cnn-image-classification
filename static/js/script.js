'use strict';
const form = document.getElementById('predictionForm');
const selector = document.getElementById('model');
const input = document.getElementById('file');
const button = document.getElementById('submit');
const statusText = document.getElementById('status');
const preview = document.getElementById('preview');
const scores = document.getElementById('scores');
let previewURL;
let requestVersion = 0;

function clearPrediction() {
  requestVersion += 1;
  scores.replaceChildren();
  document.getElementById('resultLabel').textContent = 'Choose a model and an image to begin.';
}
selector.addEventListener('change', clearPrediction);
input.addEventListener('change', () => {
  clearPrediction();
  if (previewURL) URL.revokeObjectURL(previewURL);
  preview.hidden = !input.files.length;
  if (input.files.length) {previewURL = URL.createObjectURL(input.files[0]); preview.src = previewURL;}
});
async function init() {
  try {
    const response = await fetch('/models');
    if (!response.ok) throw new Error('Could not load the model list.');
    const models = await response.json();
    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.id; option.textContent = `${model.name} · ${model.dataset}`; selector.append(option);
    }
    button.disabled = models.length === 0;
    statusText.textContent = models.length ? 'Ready for an image.' : 'No trained models are available yet.';
  } catch (error) {statusText.textContent = error.message;}
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const version = ++requestVersion;
  scores.replaceChildren();
  document.getElementById('resultLabel').textContent = 'Waiting for prediction…';
  button.disabled = true; statusText.textContent = 'Classifying…';
  try {
    const response = await fetch('/predict', {method:'POST', body:new FormData(form)});
    const data = await response.json();
    if (version !== requestVersion) return;
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    document.getElementById('resultLabel').textContent = `${data.class} · ${data.confidence.toFixed(1)}%`;
    data.all_classes.map((label,i) => ({label,score:data.all_confidences[i]}))
      .sort((a,b) => b.score-a.score).slice(0,5).forEach(item => {
        const row = document.createElement('li'); row.textContent = `${item.label} — ${item.score.toFixed(1)}%`;
        const bar = document.createElement('progress'); bar.max = 100; bar.value = item.score;
        bar.setAttribute('aria-label', item.label); row.append(bar); scores.append(row);
      });
    statusText.textContent = 'Prediction complete.';
  } catch(error) {
    if (version === requestVersion) {statusText.textContent = error.message; document.getElementById('resultLabel').textContent = 'No prediction available.';}
  } finally {button.disabled = selector.options.length === 0;}
});
init();
