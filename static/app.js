(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const fileInput = $("file"), drop = $("drop"), goBtn = $("go"), pickBtn = $("pick");
  let selected = null, data = null, activeStage = 0;

  /* ---------------- model status banner ---------------- */
  function setTrainingPanel(expanded, modelReady) {
    const content = $("trainContent"), toggle = $("trainToggle");
    if (!content || !toggle) return;
    content.classList.toggle("hidden", !expanded);
    toggle.classList.toggle("hidden", !modelReady);
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.textContent = expanded ? "Hide training options" : "Manage training / Retrain";
  }

  function loadModelStatus(collapseWhenReady) {
    return fetch("/api/status").then((r) => r.json()).then((s) => {
      const b = $("modelBanner");
      $("modelClassCount").textContent = s.model_ready ? String(s.classes.length) : "Model not available";
      $("modelAccuracy").textContent = s.validation_accuracy != null
        ? (s.validation_accuracy * 100).toFixed(2) + "%"
        : "Not available";
      $("validationContext").textContent = s.validation_note ||
        "This is a held-out validation score, not an independent test-set result or a guarantee for new photos.";
      $("validationCaveat").textContent = s.validation_note ||
        "The validation score is not an independent test result. Differences in camera, lighting, background, and plant varieties may reduce performance on new photos.";
      b.classList.remove("hidden", "ok");
      if (s.model_ready) {
        b.classList.add("ok");
        b.textContent = "Trained model loaded: " + s.classes.length + " classes ready.";
        if (s.validation_accuracy != null) {
          b.textContent += " Saved validation score: " + (s.validation_accuracy * 100).toFixed(1) + "%.";
        }
        if (s.validation_note) b.textContent += " " + s.validation_note;
        setTrainingPanel(!collapseWhenReady, true);
      } else {
        b.textContent = (s.message || "Disease classification model not available. Please train/load the model.") + " Use the Train Model section below.";
        setTrainingPanel(true, false);
      }
    }).catch(() => {
      $("modelClassCount").textContent = "Status unavailable";
      $("modelAccuracy").textContent = "Status unavailable";
      $("validationContext").textContent = "Could not retrieve validation details from the application.";
      $("validationCaveat").textContent = "Model status could not be loaded. Please check that the application server is running.";
    });
  }
  loadModelStatus(true);
  const trainToggle = $("trainToggle");
  if (trainToggle) trainToggle.addEventListener("click", () => {
    const expanded = $("trainToggle").getAttribute("aria-expanded") !== "true";
    setTrainingPanel(expanded, true);
  });
  const trainLink = document.querySelector('.topnav a[href="#train"]');
  if (trainLink) trainLink.addEventListener("click", () => {
    const trainContent = $("trainContent");
    if (trainContent && trainContent.classList.contains("hidden")) setTrainingPanel(true, true);
  });

  /* ---------------- upload handling ---------------- */
  function showError(msg) { const e = $("errorBox"); e.textContent = msg; e.classList.toggle("hidden", !msg); }

  function setFile(f) {
    showError("");
    if (!f) return;
    if (!/\.(jpe?g|png)$/i.test(f.name)) { showError("Unsupported file format. Please upload a JPG, JPEG or PNG image."); return; }
    if (f.size > 10 * 1024 * 1024) { showError("Image is too large. Maximum size is 10 MB."); return; }
    selected = f;
    $("previewImg").src = URL.createObjectURL(f);
    $("previewName").textContent = f.name;
    $("preview").classList.remove("hidden");
    goBtn.classList.remove("hidden");
  }

  pickBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  goBtn.addEventListener("click", async () => {
    showError("");
    if (!selected) { showError("No image uploaded. Please choose a leaf image."); return; }
    const fd = new FormData(); fd.append("image", selected);
    $("loading").classList.remove("hidden"); $("results").classList.add("hidden"); goBtn.disabled = true;
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      let json = null;
      try { json = await res.json(); } catch (_) {}
      if (!res.ok || !json || json.error) throw new Error((json && json.error) || "Server error. Please try again.");
      data = json; render();
    } catch (err) {
      showError(err.message || "Could not analyse the image.");
    } finally {
      $("loading").classList.add("hidden"); goBtn.disabled = false;
    }
  });


  /* ---------------- training on the website ---------------- */
  let training = false, pollTimer = null;
  const trainErr = (m) => { const e = $("trainError"); e.textContent = m || ""; e.classList.toggle("hidden", !m); };

  function renderDataset(ds) {
    const box = $("datasetList");
    $("classHints").innerHTML = ds.classes.map((c) => `<option value="${esc(c.name)}">`).join("");
    if (!ds.classes.length) { box.innerHTML = '<p class="muted small">No images uploaded yet.</p>'; return; }
    box.innerHTML = ds.classes.map((c) => `<div class="ds-row"><span>${esc(c.label)}<span class="n">${c.count} images</span>${c.count < 10 ? '<div class="ds-warn">Few images – accuracy improves with 20+ per class.</div>' : ""}</span>
      <button title="Remove class" data-c="${esc(c.name)}" ${training ? "disabled" : ""}>✕</button></div>`).join("") +
      `<div class="small muted" style="padding:8px 0">${ds.classes.length} classes · ${ds.total} images</div>`;
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm("Remove this class and its images?")) return;
      const r = await fetch("/api/train/class/" + encodeURIComponent(b.dataset.c), { method: "DELETE" });
      const j = await r.json(); if (j.error) trainErr(j.error); else renderDataset(j);
    }));
  }
  const refreshDataset = () => fetch("/api/train/dataset").then((r) => r.json()).then(renderDataset).catch(() => {});
  refreshDataset();

  async function uploadGroups(groups) {            // groups: {className: [File,...]}
    trainErr("");
    const total = Object.values(groups).reduce((n, a) => n + a.length, 0);
    if (!total) { trainErr("No JPG/PNG images were found in your selection."); return; }
    let done = 0, skipped = 0;
    $("trainBtn").disabled = true;
    try {
      for (const [cls, files] of Object.entries(groups)) {
        for (let i = 0; i < files.length; i += 10) {
          const fd = new FormData();
          fd.append("class_name", cls);
          files.slice(i, i + 10).forEach((f) => fd.append("images", f));
          const res = await fetch("/api/train/upload", { method: "POST", body: fd });
          let j = null; try { j = await res.json(); } catch (_) {}
          if (!res.ok || !j || j.error) throw new Error((j && j.error) || "Upload failed.");
          done += j.saved; skipped += j.skipped; renderDataset(j);
          $("uploadStatus").textContent = `Uploading… ${done + skipped} / ${total}`;
        }
      }
      $("uploadStatus").textContent = `Added ${done} image(s)` + (skipped ? `, skipped ${skipped} invalid/too-large file(s).` : ".");
    } catch (e) { trainErr(e.message); }
    finally { $("trainBtn").disabled = training; }
  }

  $("folderBtn").addEventListener("click", () => $("folderInput").click());
  $("folderInput").addEventListener("change", () => {
    const groups = {};
    Array.from($("folderInput").files).forEach((f) => {
      if (!/\.(jpe?g|png)$/i.test(f.name)) return;
      const parts = (f.webkitRelativePath || "").split("/");
      if (parts.length < 2) return;
      const cls = parts[parts.length - 2];
      (groups[cls] = groups[cls] || []).push(f);
    });
    $("folderInput").value = "";
    uploadGroups(groups);
  });
  $("classBtn").addEventListener("click", () => {
    if (!$("className").value.trim()) { trainErr("Please type the class name first (e.g. Healthy)."); $("className").focus(); return; }
    trainErr(""); $("classFiles").click();
  });
  $("classFiles").addEventListener("change", () => {
    const files = Array.from($("classFiles").files);
    $("classFiles").value = "";
    uploadGroups({ [$("className").value.trim()]: files });
  });

  function showTrain(st) {
    const prog = $("trainProgress"), msg = $("trainMsg");
    training = st.state === "running";
    $("trainBtn").disabled = training; $("trainBtn").textContent = training ? "Training…" : "Train Model";
    $("folderBtn").disabled = $("classBtn").disabled = training;
    if (training || st.state === "done") {
      prog.classList.remove("hidden");
      const pct = st.state === "done" ? 100 : Math.min(99, Math.round(100 * (st.epoch || 0) / Math.max(1, st.total_epochs)));
      $("trainBar").style.width = pct + "%";
      $("trainText").textContent = training
        ? `${st.phase || "Training"} – epoch ${st.epoch}/${st.total_epochs}` + (st.val_accuracy != null ? ` · train acc ${st.accuracy}% · val acc ${st.val_accuracy}%` : "")
        : "Finished";
    } else prog.classList.add("hidden");
    msg.classList.toggle("hidden", !(st.message || st.warning) );
    msg.classList.remove("ok", "error");
    if (st.state === "done") msg.classList.add("ok"); else if (st.state === "error") msg.classList.add("error");
    msg.textContent = [st.message, st.warning].filter(Boolean).join(" ");
  }

  function poll() {
    fetch("/api/train/status").then((r) => r.json()).then((st) => {
      showTrain(st);
      if (st.state === "running") pollTimer = setTimeout(poll, 2000);
      else { pollTimer = null; refreshDataset(); if (st.state === "done") loadModelStatus(); }
    }).catch(() => { pollTimer = setTimeout(poll, 4000); });
  }
  poll();

  $("trainBtn").addEventListener("click", async () => {
    trainErr("");
    const res = await fetch("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: +$("epochs").value || 8 }) });
    let j = null; try { j = await res.json(); } catch (_) {}
    if (!res.ok || !j || j.error) { trainErr((j && j.error) || "Could not start training."); return; }
    training = true; poll();
  });

  /* ---------------- stage definitions ---------------- */
  const fig = (src, cap) => `<figure class="fig"><img src="${src}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>`;

  function stages() {
    const im = data.images, a = data.acquisition, d = data.disease;
    const dState = d.status === "unavailable" ? "warn" : "";
    return [
      { name: "Image Acquisition", icon: "✓", cls: "", role: "Input / output", roleClass: "output", title: "Image Acquisition",
        desc: "The image is decoded, checked, measured, and resized if it is too large. BGR/RGB conversion prepares the image for OpenCV and the classifier.",
        stats: [["Width", a.width + " px"], ["Height", a.height + " px"], ["Channels", a.channels], ["Min intensity", a.min], ["Max intensity", a.max], ["Mean intensity", a.mean]],
        figs: [[im.original, "Original image"], [im.gray, "Grayscale"], [im.hsv, "HSV colour space"]] },
      { name: "Enhancement", icon: "✓", cls: "", role: "Classifier input", roleClass: "used", title: "Image Enhancement",
        desc: "Contrast stretching is used for classifier input. Negative and gamma correction are shown as demonstrations only.",
        figs: [[im.negative, "Image Negative"], [im.contrast, "Contrast Stretching"], [im.gamma, "Gamma Transformation (γ = 1.3)"]] },
      { name: "Filtering", icon: "✓", cls: "", role: "Classifier input", roleClass: "used", title: "Filtering / Sharpening",
        desc: "A 3 × 3 Gaussian blur is used for classifier input. Median, Laplacian and high-boost outputs are demonstrations only.",
        figs: [[im.gaussian, "Gaussian Filter"], [im.median, "Median Filter"], [im.laplacian, "Laplacian"], [im.highboost, "High-Boost Filtering"]] },
      { name: "Edge Detection", icon: "✓", cls: "", role: "Demonstration only", roleClass: "demo", title: "Edge Detection",
        desc: "Roberts, Prewitt, Sobel and Laplacian outputs demonstrate edge operators; these edge maps are not classifier inputs.",
        figs: [[im.roberts, "Roberts"], [im.prewitt, "Prewitt"], [im.sobel, "Sobel"], [im.laplacian_edge, "Laplacian"]] },
      { name: "Thresholding", icon: "✓", cls: "", role: "Mask / demo", roleClass: "mask", title: "Thresholding",
        desc: "Global, grayscale Otsu and adaptive outputs are demonstrations. Separately, Otsu thresholding on HSV saturation estimates the leaf mask for the approximate overlay.",
        stats: [["Otsu threshold", data.otsu], ["Global threshold", 120]],
        figs: [[im.global, "Global Thresholding (T = 120)"], [im.otsu, "Otsu Thresholding"], [im.adaptive, "Adaptive Thresholding"]] },
      { name: "Morphology", icon: "✓", cls: "", role: "Mask / demo", roleClass: "mask", title: "Morphological Processing",
        desc: "Dashboard examples demonstrate erosion, dilation, opening and closing. In the separate overlay pipeline, morphology and connected-component area filtering clean candidate masks; they do not affect the predicted class.",
        figs: [[im.erosion, "Erosion"], [im.dilation, "Dilation"], [im.opening, "Opening"], [im.closing, "Closing"]] },
      { name: "Disease Detection", icon: d.status === "unavailable" ? "⚠" : "🔬", cls: "disease " + dState, role: "Machine learning", roleClass: "ml", title: "🔬 Disease Detection",
        desc: "The enhanced + filtered image is passed to a trained classifier. The colour mask is experimental and does not confirm disease lesions.",
        custom: () => diseaseStage(),
        figs: [[im.classifier_input, "Processed image given to classifier"], [im.leaf_mask, "Estimated foreground mask"], [im.lesion_mask, "Experimental colour mask (not lesion detection)"]] },
      { name: "Final Result", icon: "⭐", cls: "final", role: "Output", roleClass: "output", title: "Final Result",
        desc: "Annotated leaf with the model's prediction.",
        figs: [[im.final, "Final annotated image"]], scroll: true },
    ];
  }

  function diseaseStage() {
    const d = data.disease;
    if (d.status === "unavailable")
      return `<div class="note">${esc(d.message)}<br>No disease is reported because no classifier is available.</div>`;
    return `<div class="stats"><div class="stat"><small>Status</small><b>${d.status === "healthy" ? "Healthy" : "Disease Detected"}</b></div>
      <div class="stat"><small>Prediction</small><b>${esc(d.name)}</b></div>
      <div class="stat"><small>Confidence</small><b>${d.confidence.toFixed(1)}%</b></div></div>`;
  }

  /* ---------------- rendering ---------------- */
  function render() {
    $("results").classList.remove("hidden");
    renderTimeline(); selectStage(0, false);
    renderResult(); renderInfo(); renderCompare();
    $("dashboard").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderTimeline() {
    $("timeline").innerHTML = stages().map((s, i) =>
      `<button class="stage ${s.cls}" data-i="${i}" role="tab" aria-label="${esc(s.name)}: ${esc(s.role)}"><span class="num">${i + 1}</span><span class="name">${esc(s.name)}</span><span class="stage-tag ${s.roleClass}">${esc(s.role)}</span><span class="tick">${s.icon}</span></button>`).join("");
    $("timeline").querySelectorAll(".stage").forEach((b) => b.addEventListener("click", () => selectStage(+b.dataset.i, true)));
  }

  function selectStage(i, scroll) {
    activeStage = i;
    const s = stages()[i];
    document.querySelectorAll(".stage").forEach((b, k) => b.classList.toggle("active", k === i));
    let html = `<h3>${i + 1}. ${esc(s.title)}</h3><span class="stage-tag ${s.roleClass}">${esc(s.role)}</span><p class="desc">${esc(s.desc)}</p>`;
    if (s.stats) html += `<div class="stats">${s.stats.map((x) => `<div class="stat"><small>${esc(x[0])}</small><b>${esc(x[1])}</b></div>`).join("")}</div>`;
    if (s.custom) html += s.custom();
    html += `<div class="gallery">${s.figs.map((f) => fig(f[0], f[1])).join("")}</div>`;
    $("stageView").innerHTML = html;
    bindZoom($("stageView"));
    if (s.scroll && scroll) $("resultSection").scrollIntoView({ behavior: "smooth" });
  }

  function renderResult() {
    const d = data.disease, card = $("resultCard");
    if (d.status === "unavailable") {
      card.className = "result-card unavailable";
      card.innerHTML = `<div class="result-banner"><div class="status">Final Disease Detection</div><h3>Model Not Available</h3></div>
        <div class="unavail-msg"><p><b>${esc(d.message)}</b></p>
        <p class="muted">The DIP stages above still ran normally, but segmentation alone is not disease classification, so no disease name or confidence is shown.</p>
        <p class="muted">To enable detection: scroll to <b>Train the Disease Classifier</b> at the top of this page, upload your labelled leaf images and press <b>Train Model</b>.</p></div>`;
      return;
    }
    const healthy = d.status === "healthy";
    card.className = "result-card " + d.status;
    const top = (d.top || []).map((t) => `<li><span>${esc(t.name)}</span><b>${t.confidence.toFixed(1)}%</b></li>`).join("");
    card.innerHTML = `
      <div class="result-banner">
        <div class="status">${healthy ? "Healthy" : "Disease Detected"}</div>
        <h3>${healthy ? "Healthy Leaf" : esc(d.name)}</h3>
        <div class="conf">Confidence: ${d.confidence.toFixed(1)}%</div>
      </div>
      <div class="result-body">
        <div>
          <div class="final-img"><img id="finalImg" src="${d.final_image}" alt="Final annotated leaf" style="cursor:zoom-in"></div>
          <p class="legend-note">${healthy ? "Leaf shown without any disease marking." : d.region.marked ? "Red overlay/outline = colour-based candidate area; not a model-confirmed lesion." : "No region marked because the leaf could not be isolated reliably."}</p>
        </div>
        <div class="details">
          <div class="detail-block"><h4>Status</h4><div>${healthy ? "Healthy" : "Disease Detected"}</div></div>
          <div class="detail-block"><h4>${healthy ? "Result" : "Disease"}</h4><div>${healthy ? "Healthy Leaf" : esc(d.name)}</div></div>
          <div class="detail-block"><h4>Model confidence score</h4><div class="meter"><span style="width:${Math.min(100, d.confidence)}%"></span></div>
            <div>${d.confidence.toFixed(1)}% (not overall model accuracy)</div>
            ${d.low_confidence ? '<div class="low">⚠ Low confidence – the prediction may be unreliable. Try a clearer, well-lit photo.</div>' : ""}</div>
          <div class="detail-block"><h4>Experimental colour estimate</h4><div>${esc(d.region.text)}</div></div>
          <div class="detail-block"><h4>About this result</h4><div>${esc(d.info.symptoms)}</div></div>
          <div class="detail-block"><h4>Top predictions</h4><ul class="top-list">${top}</ul></div>
          <p class="muted">The classifier always chooses among its trained classes. An unrelated plant or a different disease may be misclassified.</p>
          <a class="btn download" href="${d.download_url}" download="leafvision_final_result.png">⬇ Download Final Result</a>
        </div>
      </div>`;
    $("finalImg").addEventListener("click", () => zoom(d.final_image));
  }

  function renderInfo() {
    const d = data.disease, box = $("infoCard");
    if (d.status === "unavailable") {
      box.innerHTML = `<p class="muted">Disease information appears here after a trained model makes a prediction.</p>
        <div class="disclaimer">${esc(data.disclaimer)}</div>`;
      return;
    }
    const i = d.info;
    box.innerHTML = `
      <div class="info-grid">
        <div class="info-box"><h4>Disease name</h4><p>${esc(i.name)}</p></div>
        <div class="info-box"><h4>Common symptoms</h4><p>${esc(i.symptoms)}</p></div>
        <div class="info-box"><h4>Cause / pathogen</h4><p>${esc(i.cause)}</p></div>
        <div class="info-box"><h4>General management / prevention</h4><ul>${i.management.map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div>
      </div>
      ${i.known ? "" : '<p class="muted">Detailed information for this class is not stored in the project, so general advice is shown.</p>'}
      <div class="disclaimer">${esc(i.disclaimer)}</div>`;
  }

  function renderCompare() {
    const im = data.images;
    const items = [["Original", im.original], ["Enhanced", im.contrast], ["Filtered", im.gaussian],
      ["Segmented", im.otsu], ["Morphology", im.morph_clean], ["Final Result", im.final]];
    $("compareCards").innerHTML = items.map((x) => fig(x[1], x[0])).join("");
    bindZoom($("compareCards"));
    const sel = $("compareSelect");
    sel.innerHTML = items.slice(1).map((x, k) => `<option value="${k + 1}">${x[0]}</option>`).join("");
    sel.onchange = setSlider;
    $("baBeforeImg").src = im.original;
    $("baRange").value = 50; setSlider(); moveSlider();
    function setSlider() { $("baAfter").src = items[+sel.value][1]; }
  }

  function moveSlider() {
    const v = $("baRange").value;
    $("baBefore").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    $("baLine").style.left = v + "%";
  }
  $("baRange").addEventListener("input", moveSlider);

  /* ---------------- image zoom ---------------- */
  function zoom(src) {
    const l = document.createElement("div"); l.className = "lightbox";
    l.innerHTML = `<img src="${src}" alt="Enlarged result">`;
    l.addEventListener("click", () => l.remove());
    document.body.appendChild(l);
  }
  function bindZoom(root) { root.querySelectorAll(".fig img").forEach((im) => im.addEventListener("click", () => zoom(im.src))); }
})();
