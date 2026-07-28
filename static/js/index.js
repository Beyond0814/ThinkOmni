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

      const details = element("details");
      details.append(element("summary", "", "Forensic reasoning - complete TXT response"));
      const response = element("pre", "reasoning-text");
      response.textContent = sample.response;
      details.append(response);
      card.append(details);
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

    fetch("static/data/samples.json")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        samples = data;
        [...new Set(samples.map((sample) => sample.dataset))].sort().forEach((dataset) => {
          const option = element("option", "", dataset);
          option.value = dataset;
          datasetSelect.append(option);
        });
        filtered = samples;
        render();
      })
      .catch(() => {
        summary.textContent = "The sample library could not be loaded.";
      });
  }
});
