document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) window.lucide.createIcons();

  const toggle = document.querySelector(".nav-toggle");
  const menu = document.querySelector(".nav-menu");
  toggle?.addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  menu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
    menu.classList.remove("open");
    toggle.setAttribute("aria-expanded", "false");
  }));

  const copyButton = document.querySelector(".copy-button");
  copyButton?.addEventListener("click", async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    if (!target) return;
    await navigator.clipboard.writeText(target.innerText);
    const label = copyButton.querySelector("span");
    label.textContent = "Copied";
    window.setTimeout(() => { label.textContent = "Copy"; }, 1600);
  });

  const sections = [...document.querySelectorAll("main section[id]")];
  const links = [...document.querySelectorAll(".nav-menu > a[href^='#']")];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
    });
  }, { rootMargin: "-35% 0px -55%", threshold: 0 });
  sections.forEach((section) => observer.observe(section));

  const sampleGrid = document.getElementById("sample-grid");
  if (sampleGrid) {
    const datasetSelect = document.getElementById("sample-dataset");
    const classSelect = document.getElementById("sample-class");
    const statusSelect = document.getElementById("sample-status-filter");
    const searchInput = document.getElementById("sample-search");
    const summary = document.getElementById("sample-summary");
    const previous = document.getElementById("sample-previous");
    const next = document.getElementById("sample-next");
    const pageLabel = document.getElementById("sample-page");
    const pageSize = 6;
    let samples = [];
    let filtered = [];
    let page = 1;

    const sampleRoot = "static/samples/";
    const unmappedCanonicalTextFiles = [
      "AV-Deepfake1M-PlusPlus/real_p1_fd28b30672.txt",
      "ArEnAV/real_p1_96fc781908.txt",
      "ArEnAV/real_video_fake_audio_p1_13c4f8ad0e.txt",
      "ArEnAV/real_video_fake_audio_p1_9cadac59e8.txt",
    ];
    const classNames = { "0": "Fully real", "1": "Fully fake", "2": "Partially fake" };
    const detectionResult = /Detection Result:\s*([^\n\r]+)/g;
    const localizationResult = /Localization Result:\s*([^\n\r]+)/g;

    const lastField = (pattern, text) => {
      const matches = [...String(text || "").matchAll(pattern)];
      return matches.length ? matches[matches.length - 1][1].trim() : "Unavailable";
    };

    const resultFromText = (text) => {
      const classId = lastField(detectionResult, text);
      return {
        classId,
        className: classNames[classId] || classId,
        localization: lastField(localizationResult, text),
      };
    };

    const sourceUrl = (relativePath) => encodeURI(`${sampleRoot}${relativePath}`);

    const readSourceSample = async (textPath) => {
      const response = await fetch(sourceUrl(textPath));
      if (!response.ok) throw new Error(`${textPath}: HTTP ${response.status}`);
      const data = await response.json();
      const slash = textPath.lastIndexOf("/");
      const directory = textPath.slice(0, slash);
      const sample = textPath.slice(slash + 1, -4);
      const audioReference = String(data.audios?.[0] || "");
      const audioExtension = audioReference.slice(audioReference.lastIndexOf(".")).toLowerCase();
      const imageReference = String(data.images?.[0]?.path || "");
      const imageName = imageReference.split(/[\\/]/).pop();
      const audioPath = `${directory}/${sample}${audioExtension}`;
      const imagePath = `${directory}/${imageName}`;
      return {
        id: `${directory}/${sample}`,
        dataset: directory,
        sample,
        audio: sourceUrl(audioPath),
        image: sourceUrl(imagePath),
        groundTruth: resultFromText(data.labels),
        prediction: resultFromText(data.response),
        response: data.response || "",
      };
    };

    const isCorrect = (sample) => sample.groundTruth.classId === sample.prediction.classId
      && sample.groundTruth.localization === sample.prediction.localization;

    const element = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };

    const resultPanel = (label, result) => {
      const panel = element("div", "sample-result");
      panel.append(element("span", "sample-result-label", label));
      const value = element("strong", "sample-result-value", result.className);
      value.append(element("small", "", `Class ${result.classId}`));
      panel.append(value);
      panel.append(element("p", "", `Localization: ${result.localization}`));
      return panel;
    };

    const sampleCard = (sample, index) => {
      const card = element("article", "sample-card");
      const header = element("header");
      header.append(element("span", "sample-index", String(index + 1).padStart(2, "0")));
      const heading = element("div");
      heading.append(element("p", "sample-dataset", sample.dataset));
      heading.append(element("h3", "", sample.groundTruth.className));
      heading.append(element("p", "sample-id", sample.sample));
      header.append(heading);
      card.append(header);

      const image = element("img", "sample-spectrogram");
      image.src = sample.image;
      image.alt = `Spectrogram of ${sample.sample} from ${sample.dataset}`;
      image.loading = "lazy";
      card.append(image);

      const audio = element("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = sample.audio;
      card.append(audio);

      const results = element("div", "sample-results");
      results.append(resultPanel("Ground truth", sample.groundTruth));
      results.append(resultPanel("ThinkOmni", sample.prediction));
      card.append(results);

      const reasoning = element("section", "sample-reasoning");
      reasoning.append(element("h4", "", "Forensic reasoning"));
      const response = element("pre", "reasoning-text");
      response.textContent = sample.response;
      reasoning.append(response);
      card.append(reasoning);
      return card;
    };

    const render = () => {
      const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.min(page, pages);
      const start = (page - 1) * pageSize;
      const current = filtered.slice(start, start + pageSize);
      sampleGrid.replaceChildren(...current.map((sample, index) => sampleCard(sample, start + index)));
      summary.textContent = `${filtered.length} of ${samples.length} samples shown`;
      pageLabel.textContent = `Page ${page} of ${pages}`;
      previous.disabled = page <= 1;
      next.disabled = page >= pages;
      if (!current.length) sampleGrid.append(element("p", "sample-empty", "No samples match the selected filters."));
    };

    const filter = () => {
      const dataset = datasetSelect.value;
      const classId = classSelect.value;
      const status = statusSelect.value;
      const query = searchInput.value.trim().toLowerCase();
      filtered = samples.filter((sample) => {
        if (dataset !== "all" && sample.dataset !== dataset) return false;
        if (classId !== "all" && sample.groundTruth.classId !== classId) return false;
        if (status === "correct" && !isCorrect(sample)) return false;
        if (status === "mismatch" && isCorrect(sample)) return false;
        if (query) {
          const searchable = `${sample.dataset} ${sample.sample} ${sample.response}`.toLowerCase();
          if (!searchable.includes(query)) return false;
        }
        return true;
      });
      page = 1;
      render();
    };

    [datasetSelect, classSelect, statusSelect].forEach((control) => control.addEventListener("change", filter));
    searchInput.addEventListener("input", filter);
    previous.addEventListener("click", () => { page -= 1; render(); });
    next.addEventListener("click", () => { page += 1; render(); });

    fetch(`${sampleRoot}sample_rename_manifest_20260728.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((manifest) => {
        const renamedTextFiles = manifest.renames
          .filter((item) => item.new.toLowerCase().endsWith(".txt"))
          .map((item) => item.new);
        const canonicalTextFiles = [...new Set([...renamedTextFiles, ...unmappedCanonicalTextFiles])];
        if (canonicalTextFiles.length !== manifest.sample_count) {
          throw new Error(`Expected ${manifest.sample_count} canonical samples, found ${canonicalTextFiles.length}`);
        }
        return Promise.all(canonicalTextFiles.map(readSourceSample));
      })
      .then((data) => {
        samples = data.sort((a, b) => a.id.localeCompare(b.id));
        [...new Set(samples.map((sample) => sample.dataset))].sort().forEach((dataset) => {
          const option = element("option", "", dataset);
          option.value = dataset;
          datasetSelect.append(option);
        });
        filtered = samples;
        render();
      })
      .catch((error) => {
        console.error(error);
        summary.textContent = "The source sample library could not be loaded.";
      });
  }
});
