const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  files: [],
  midiStems: new Set(),
  polling: new Map(),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

/* ---------- setup ---------- */

async function init() {
  state.config = await api("/api/config");
  const { models, device, default_output, default_midi_stems, daw, native_pickers } = state.config;

  $("device-badge").textContent =
    device === "mps" ? "Apple Silicon GPU" : device === "cuda" ? "CUDA GPU" : "CPU";

  const modelSelect = $("model");
  for (const [name, info] of Object.entries(models)) {
    const option = new Option(info.label, name, info.default, info.default);
    modelSelect.add(option);
  }
  modelSelect.addEventListener("change", onModelChange);

  default_midi_stems.forEach((s) => state.midiStems.add(s));
  onModelChange();

  $("output").value = default_output;
  $("browse").hidden = !native_pickers;
  $("pick-output").hidden = !native_pickers;

  for (const button of document.querySelectorAll('[data-action="logic"]')) {
    button.dataset.available = daw.logic;
  }

  $("browse").addEventListener("click", async () => {
    const { paths } = await api("/api/pick/files", { method: "POST" });
    addFiles(paths);
  });

  $("pick-output").addEventListener("click", async () => {
    const { path } = await api("/api/pick/folder", { method: "POST" });
    if (path) $("output").value = path;
  });

  $("shifts").addEventListener("input", (event) => {
    const n = Number(event.target.value);
    $("shifts-value").textContent = n === 1 ? "1 pass" : `${n} passes`;
  });

  $("quantize").addEventListener("change", (event) => {
    const on = event.target.value !== "0";
    $("strength-field").hidden = !on;
    $("quantize-value").textContent = on
      ? event.target.selectedOptions[0].textContent.split(" —")[0]
      : "Off";
  });

  $("quantize-strength").addEventListener("input", (event) => {
    $("strength-value").textContent = `${event.target.value}%`;
  });

  $("run").addEventListener("click", startJobs);
  setupDropZone();
}

function onModelChange() {
  const info = state.config.models[$("model").value];
  $("model-hint").textContent = `${info.stems.join(", ")} — ${info.notes}`;
  renderMidiChips(info.stems);
}

function renderMidiChips(stems) {
  const host = $("midi-stems");
  host.replaceChildren();
  for (const stem of stems) {
    if (stem === "drums") continue; // drums have their own toggle
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = stem;
    chip.setAttribute("aria-pressed", state.midiStems.has(stem));
    chip.addEventListener("click", () => {
      const on = chip.getAttribute("aria-pressed") === "true";
      chip.setAttribute("aria-pressed", !on);
      on ? state.midiStems.delete(stem) : state.midiStems.add(stem);
    });
    host.append(chip);
  }
}

/* ---------- file queue ---------- */

function setupDropZone() {
  const zone = $("drop");
  const stop = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      stop(event);
      zone.classList.add("over");
    })
  );

  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      stop(event);
      zone.classList.remove("over");
    })
  );

  zone.addEventListener("drop", async (event) => {
    // Finder drags sometimes carry a file:// URI list, which lets us read the
    // file in place. When they do not, fall back to uploading the bytes to the
    // local server — either way nothing leaves this machine.
    const uris = event.dataTransfer.getData("text/uri-list");
    const paths = uris
      .split(/\r?\n/)
      .filter((line) => line.startsWith("file://"))
      .map((line) => decodeURIComponent(line.replace("file://", "")));

    if (paths.length) {
      addFiles(paths);
      return;
    }

    const dropped = [...event.dataTransfer.files];
    if (!dropped.length) return;

    const zoneTitle = zone.querySelector(".drop-title");
    const original = zoneTitle.textContent;
    zoneTitle.textContent = `Reading ${dropped.length} file${dropped.length > 1 ? "s" : ""}…`;
    try {
      const form = new FormData();
      dropped.forEach((file) => form.append("files", file));
      const response = await fetch("/api/upload", { method: "POST", body: form });
      if (!response.ok) throw new Error(await response.text());
      addFiles((await response.json()).paths);
    } catch (error) {
      alert(`Could not read the dropped files: ${error.message}`);
    } finally {
      zoneTitle.textContent = original;
    }
  });
}

function addFiles(paths) {
  for (const path of paths) {
    if (path && !state.files.includes(path)) state.files.push(path);
  }
  renderQueue();
}

function renderQueue() {
  const list = $("queue");
  list.replaceChildren();

  for (const path of state.files) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = path.split("/").pop();
    name.title = path;

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      state.files = state.files.filter((p) => p !== path);
      renderQueue();
    });

    item.append(name, remove);
    list.append(item);
  }

  $("run").disabled = state.files.length === 0;
  $("run").textContent =
    state.files.length > 1 ? `Process ${state.files.length} files` : "Process";
}

/* ---------- running ---------- */

async function startJobs() {
  const payload = {
    paths: state.files,
    output_dir: $("output").value,
    model: $("model").value,
    shifts: Number($("shifts").value),
    midi_stems: [...state.midiStems],
    transcribe_drums: $("drum-midi").checked,
    quantize: Number($("quantize").value),
    quantize_strength: Number($("quantize-strength").value) / 100,
    bit_depth: $("bit-depth").value,
    include_residual: $("residual").checked,
    normalize_stems: $("normalize").checked,
  };

  $("run").disabled = true;
  try {
    const { jobs } = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.files = [];
    renderQueue();
    jobs.forEach(track);
  } catch (error) {
    alert(error.message);
    $("run").disabled = false;
  }
}

function cardFor(job) {
  let card = document.querySelector(`[data-job="${job.id}"]`);
  if (card) return card;

  card = $("result-template").content.firstElementChild.cloneNode(true);
  card.dataset.job = job.id;
  card.querySelector(".result-name").textContent = job.source_name;
  $("results").prepend(card);

  card.querySelectorAll(".actions button").forEach((button) => {
    if (button.dataset.available === "false") button.disabled = true;
    button.addEventListener("click", async () => {
      try {
        await api(`/api/jobs/${job.id}/open`, {
          method: "POST",
          body: JSON.stringify({
            target: button.dataset.action,
            what: button.dataset.what || "folder",
          }),
        });
      } catch (error) {
        alert(error.message);
      }
    });
  });

  return card;
}

function track(job) {
  render(job);
  if (job.status === "done" || job.status === "error") return;
  const timer = setInterval(async () => {
    try {
      const latest = await api(`/api/jobs/${job.id}`);
      render(latest);
      if (latest.status === "done" || latest.status === "error") {
        clearInterval(timer);
        state.polling.delete(job.id);
      }
    } catch {
      clearInterval(timer);
    }
  }, 700);
  state.polling.set(job.id, timer);
}

function render(job) {
  const card = cardFor(job);
  card.querySelector(".result-status").textContent = job.message;
  card.querySelector(".bar-fill").style.width = `${job.fraction * 100}%`;

  const error = card.querySelector(".error");
  error.hidden = !job.error;
  if (job.error) error.textContent = job.error;

  if (job.status !== "done" || !job.result) return;
  if (card.classList.contains("done")) return;
  card.classList.add("done");

  const { analysis, stems, midi } = job.result;
  const badges = card.querySelector(".badges");
  badges.replaceChildren(
    badge(analysis.key.name),
    badge(`${analysis.tempo.bpm.toFixed(2)} BPM`),
    badge(`Camelot ${analysis.key.camelot}`, true),
    badge(`downbeat ${analysis.tempo.first_beat.toFixed(3)}s`, true)
  );

  const tracks = card.querySelector(".tracks");
  tracks.replaceChildren();
  for (const [name, path] of Object.entries(stems)) {
    tracks.append(trackRow(name, path, midi[name]));
  }
  if (midi.all) {
    tracks.append(trackRow("all stems", null, midi.all));
  }

  card.querySelector(".result-body").hidden = false;
  if (state.polling.size === 0) $("run").disabled = state.files.length === 0;
}

function badge(text, quiet = false) {
  const span = document.createElement("span");
  span.className = quiet ? "badge quiet" : "badge";
  span.textContent = text;
  return span;
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function trackRow(name, wavPath, midiPath) {
  const row = document.createElement("div");
  row.className = "track";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = name;

  const middle = document.createElement("div");
  if (wavPath) {
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "none";
    player.src = fileUrl(wavPath);
    middle.append(player);
  }

  const links = document.createElement("div");
  links.className = "links";
  if (wavPath) links.append(link("WAV", wavPath));
  if (midiPath) links.append(link("MIDI", midiPath));

  row.append(label, middle, links);
  return row;
}

function link(text, path) {
  const anchor = document.createElement("a");
  anchor.textContent = text;
  anchor.href = fileUrl(path);
  anchor.download = path.split("/").pop();
  return anchor;
}

init().catch((error) => {
  document.body.innerHTML = `<p class="error" style="padding:40px">Could not start: ${error.message}</p>`;
});
