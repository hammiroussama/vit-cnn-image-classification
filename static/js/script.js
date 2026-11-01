'use strict';
const $ = id => document.getElementById(id);
const form = $('predictionForm'), selector = $('model'), input = $('file'), button = $('submit');
let models = [], predictions = [], showAll = false, previewURL, controller, requestVersion = 0, busy = false;
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function updateButton() { button.disabled = busy || !selector.value || !input.files.length; }
function reset() {
  requestVersion++;
  if (controller) controller.abort();
  busy = false; predictions = [];
  $('resultPanel').setAttribute('aria-busy','false');
  $('predictionContent').hidden = true; $('emptyResult').hidden = false;
  $('emptyResult').querySelector('h3').textContent = 'A little curiosity goes a long way.';
  $('emptyResult').querySelector('p').textContent = 'Your prediction and ranked class scores will appear here.';
  $('resultState').textContent = 'Awaiting prediction'; $('buttonLabel').textContent = 'Run prediction';
  $('scores').replaceChildren(); updateButton();
}
function modelChanged() {
  reset();
  const model = models.find(m => m.id === selector.value);
  const cifar = model?.dataset === 'cifar10';
  $('datasetBadge').textContent = model ? (cifar ? 'CIFAR-10' : 'Imagenette') : 'No models';
  const preliminary = /one.epoch|preliminary|smoke/i.test(model?.name || '');
  $('modelNote').textContent = preliminary ? 'Preliminary checkpoint · one epoch of training. Results are exploratory.' : 'An experimental checkpoint. See the project README for evaluation details.';
  $('classHint').textContent = !model ? 'Select a model to see its supported classes.' : cifar ? 'Airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.' : 'Tench, English springer spaniel, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball and parachute.';
  status(model ? 'Choose an image, then run a prediction.' : 'No trained models are registered yet.');
}
selector.addEventListener('change', modelChanged);
input.addEventListener('change', () => {
  reset();
  if (previewURL) URL.revokeObjectURL(previewURL);
  const file = input.files[0];
  $('preview').hidden = true; $('placeholder').hidden = false; $('imageStage').classList.remove('has-image');
  $('fileName').textContent = 'No image selected';
  if (!file) {status('Choose an image.'); return;}
  if (file.size > 5*1024*1024 || !['image/jpeg','image/png','image/webp'].includes(file.type)) {
    input.value = ''; updateButton(); status('Choose a JPEG, PNG or WebP image smaller than 5 MB.',true); return;
  }
  previewURL = URL.createObjectURL(file); $('preview').src = previewURL;
  $('preview').hidden = false; $('placeholder').hidden = true; $('imageStage').classList.add('has-image');
  $('fileName').textContent = file.name; status('Image ready. Run a prediction to explore the results.'); updateButton();
});
$('preview').addEventListener('error', () => {
  input.value = ''; $('preview').hidden = true; $('placeholder').hidden = false; $('imageStage').classList.remove('has-image');
  reset(); status('This file could not be displayed as an image. Please choose another.',true);
});
function drawScores() {
  $('topFive').setAttribute('aria-pressed',String(!showAll)); $('allClasses').setAttribute('aria-pressed',String(showAll));
  $('scores').replaceChildren();
  predictions.slice(0,showAll ? 10 : 5).forEach(item => {
    const row = document.createElement('li'), labels = document.createElement('div'), name = document.createElement('span'), value = document.createElement('strong');
    labels.className = 'bar-label'; name.textContent = item.label; value.textContent = `${item.score.toFixed(1)}%`; labels.append(name,value);
    const track = document.createElement('div'), fill = document.createElement('div'); track.className = 'bar-track'; fill.className = 'bar-fill';
    fill.style.width = `${item.score}%`; track.setAttribute('aria-hidden','true'); track.append(fill); row.append(labels,track); $('scores').append(row);
  });
  const other = predictions.slice(5).reduce((sum,item) => sum+item.score,0);
  $('remaining').textContent = showAll ? 'All ten classes shown. Bars use the same 0–100% scale.' : `Other five classes: ${other.toFixed(1)}% combined. Bars use a 0–100% scale.`;
}
$('topFive').addEventListener('click',()=>{showAll=false;drawScores();});
$('allClasses').addEventListener('click',()=>{showAll=true;drawScores();});
form.addEventListener('submit',async event => {
  event.preventDefault(); if (!input.files.length || !selector.value || busy) return;
  reset(); const version = requestVersion; controller = new AbortController(); busy = true; updateButton();
  $('resultPanel').setAttribute('aria-busy','true'); $('resultState').textContent = 'Processing'; $('buttonLabel').textContent = 'Analyzing image…';
  $('emptyResult').querySelector('h3').textContent = 'Taking a closer look…';
  $('emptyResult').querySelector('p').textContent = 'The first prediction may take longer while the model loads.';
  status('Running inference. You can choose another image to cancel this request.');
  try {
    const response = await fetch('/predict',{method:'POST',body:new FormData(form),signal:controller.signal});
    let data; try {data = await response.json();} catch {throw new Error('The server returned an unreadable response. Check that the demo is running.');}
    if (version !== requestVersion) return;
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    if (!Array.isArray(data.all_classes) || data.all_classes.length !== 10 || !Array.isArray(data.all_confidences) || data.all_confidences.length !== 10 || !data.all_confidences.every(s=>Number.isFinite(s) && s>=0 && s<=100)) throw new Error('The model returned invalid class scores.');
    predictions = data.all_classes.map((label,i)=>({label:String(label),score:data.all_confidences[i]})).sort((a,b)=>b.score-a.score);
    const top = predictions[0], runner = predictions[1];
    $('resultLabel').textContent = top.label; $('topScore').textContent = `${top.score.toFixed(1)}%`;
    $('scoreRing').style.setProperty('--score',`${top.score}%`);
    $('runnerUp').textContent = `${(top.score-runner.score).toFixed(1)} percentage points ahead of ${runner.label}.`;
    $('emptyResult').hidden = true; $('predictionContent').hidden = false; $('resultState').textContent = 'Complete'; drawScores(); status('Prediction complete. Explore the top five or all ten classes.');
  } catch(error) {
    if (version === requestVersion && error.name !== 'AbortError') {
      $('resultState').textContent = 'Try again'; $('emptyResult').querySelector('h3').textContent = 'No prediction available.';
      $('emptyResult').querySelector('p').textContent = 'Check the message below the image, then try again.'; status(error.message,true);
    }
  } finally {
    if (version === requestVersion) {busy=false;$('resultPanel').setAttribute('aria-busy','false');$('buttonLabel').textContent='Run prediction';updateButton();}
  }
});
async function init() {
  try {
    const response = await fetch('/models'); if (!response.ok) throw new Error('Could not load models. Check the server connection.');
    models = await response.json();
    for (const model of models) {const option=document.createElement('option');option.value=model.id;option.textContent=model.name;selector.append(option);}
    if (!models.length) {const option=document.createElement('option');option.value='';option.textContent='No models registered';selector.append(option);selector.disabled=true;}
    modelChanged();
  } catch(error) {status(error.message,true);}
}
init();
