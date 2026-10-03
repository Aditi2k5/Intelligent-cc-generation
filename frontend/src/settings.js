// Settings dialog: the OpenAI API key (saved by the backend in the user's
// profile, never in the app) and the optional NVIDIA GPU download.
const $ = selector => document.querySelector(selector);
const dialog = $('#settingsDialog');
const keyInput = $('#openaiKey');
const keyError = $('#keyError');
const gpuToggle = $('#gpuToggle');
let settings = null;
let pollTimer = null;

async function request(method, body) {
  const response = await fetch('/api/settings', {
    method,
    cache: 'no-store',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'Could not save the settings.');
  return data;
}

function render() {
  const keyStatus = $('#keyStatus');
  if (settings.openai_key_set) {
    keyStatus.textContent = settings.openai_key_source === 'environment'
      ? 'Using the key from the OPENAI_API_KEY environment variable. A key saved here replaces it.'
      : `Key saved (ends ${settings.openai_key_hint}). Paste a new one to replace it.`;
    keyStatus.classList.add('ok');
  } else {
    keyStatus.textContent = 'No key saved yet.';
    keyStatus.classList.remove('ok');
  }
  $('#removeKey').classList.toggle('hidden', settings.openai_key_source !== 'settings');
  $('#settingsIntro').classList.toggle('hidden', settings.openai_key_set);

  const gpu = settings.gpu;
  $('#gpuSection').classList.toggle('hidden', !gpu.supported && !gpu.enabled);
  gpuToggle.checked = gpu.enabled;
  gpuToggle.disabled = gpu.state === 'downloading';
  $('#gpuStatus').textContent = gpu.message || (gpu.enabled ? '' : 'Off: processing runs on the CPU.');
  clearTimeout(pollTimer);
  if (gpu.state === 'downloading') pollTimer = setTimeout(refresh, 2000);
}

async function refresh() {
  try {
    settings = await request('GET');
    render();
  } catch {
    pollTimer = setTimeout(refresh, 5000);
  }
}

function announce() {
  window.dispatchEvent(new CustomEvent('planetread:settings', { detail: settings }));
}

function showError(message) {
  keyError.textContent = message;
  keyError.classList.toggle('hidden', !message);
}

function open() {
  showError('');
  if (!dialog.open) dialog.showModal();
  if (!settings?.openai_key_set) keyInput.focus();
}

$('#settingsButton').addEventListener('click', open);
$('#settingsClose').addEventListener('click', () => dialog.close());

$('#settingsForm').addEventListener('submit', async event => {
  event.preventDefault();
  const key = keyInput.value.trim();
  if (!key) return showError('Paste your OpenAI API key first.');
  const button = $('#saveKey');
  button.disabled = true;
  button.textContent = 'Checking…';
  try {
    settings = await request('POST', { openai_api_key: key });
    keyInput.value = '';
    showError('');
    announce();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Save key';
  }
});

$('#removeKey').addEventListener('click', async () => {
  try {
    settings = await request('POST', { openai_api_key: '' });
    announce();
  } catch (error) {
    showError(error.message);
  }
});

gpuToggle.addEventListener('change', async () => {
  try {
    settings = await request('POST', { gpu: gpuToggle.checked });
    render();
  } catch (error) {
    $('#gpuStatus').textContent = error.message;
  }
});

// Keep the key field on the main page (index.html) in step with this dialog.
window.addEventListener('planetread:settings', event => {
  settings = event.detail;
  render();
});

refresh();
