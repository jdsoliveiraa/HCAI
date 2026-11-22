/* Realism review form (ADR 0010). Answers are a draft in localStorage; "Export review" downloads the record
   (Markdown + JSON) to commit to docs/pilot/. reviewMarkdown mirrors hcai.review.review_markdown line by line,
   and checkReview mirrors hcai.review.check_review; a test runs both sides on the same answers.

   What the page adds on top of the record is derived from it or kept beside it, never in it (the record stays
   format version 1): the progress (itemDone), the current item and the keyboard shortcuts, and whether the
   answers have changed since the last export. */
(function () {
  "use strict";

  var FORMAT = "hcai-realism-review";
  var FORMAT_VERSION = 1;

  function tick(value) { return value === true ? "x" : " "; }

  function notesLine(prefix, notes) {
    var lines = String(notes || "").trim().replace(/\r\n/g, "\n").split("\n");
    var text = lines.map(function (line) { return line.replace(/\s+$/, ""); }).join("\n  ");
    return text ? prefix + " " + text : prefix;
  }

  function reviewMarkdown(data, review) {
    var meta = data.meta;
    var answers = {};
    review.items.forEach(function (a) { answers[a.id] = a; });
    var langs = meta.languages.map(function (l) { return l.toUpperCase(); }).join(" + ");
    var cvs = data.items.filter(function (i) { return i.kind === "cv"; });
    var roles = {};
    data.items.forEach(function (i) { if (i.kind === "job_ad") { roles[i.role] = true; } });
    var out = [
      "# CV realism review (pilot, #11 Part B)", "",
      "Hand review of a stratified sample of the frozen corpus " + meta.corpus_version + " (generator " +
        meta.generator + "): " + Object.keys(roles).length + " job ads (" + langs + ") and " + cvs.length +
        " CV variants, one per role x language, drawn by hcai.review.review_sample with seed " + meta.seed + ".", "",
      "For each document, tick or note:", ""
    ];
    data.guidance.forEach(function (g) { out.push("- **" + g[0] + "**: " + g[1]); });
    out.push("",
      "Reviewer: " + (review.reviewer || "______") + "  Date: " + (review.date || "______"), "",
      "## Overall verdict", "",
      "- [" + tick(review.verdict.freeze) + "] " + data.verdict_label,
      notesLine("- Notes:", review.verdict.notes), "", "");
    data.items.forEach(function (item) {
      var answer = answers[item.id];
      var boxes = item.checks.map(function (c) {
        return "- [" + tick(answer.checks[c]) + "] " + data.labels[c];
      }).join("  ");
      out.push("## " + item.title, "");
      if (item.kind === "job_ad") {
        out.push(notesLine(boxes + "  - Notes:", answer.notes));
      } else {
        out.push(boxes, notesLine("- Notes:", answer.notes));
      }
      out.push("", "<details><summary>Text</summary>", "", item.text, "", "</details>", "");
    });
    return out.join("\n");
  }

  function checkReview(review, data) {
    var meta = data.meta;
    if (!review || typeof review !== "object" || Array.isArray(review) || review.format !== FORMAT) {
      return "not a realism review export";
    }
    if (review.format_version !== FORMAT_VERSION) {
      return "unsupported format version " + JSON.stringify(review.format_version);
    }
    var keys = ["corpus_version", "seed"];
    for (var k = 0; k < keys.length; k++) {
      if (review[keys[k]] !== meta[keys[k]]) {
        return "it reviews " + keys[k].replace("_", " ") + " " + JSON.stringify(review[keys[k]]) +
          ", this sample is " + JSON.stringify(meta[keys[k]]);
      }
    }
    var got = (Array.isArray(review.items) ? review.items : [])
      .filter(function (i) { return i && typeof i === "object"; })
      .map(function (i) { return String(i.id); }).sort();
    var want = data.items.map(function (i) { return i.id; }).sort();
    if (got.length !== want.length || got.some(function (id, n) { return id !== want[n]; })) {
      return "its items differ from this sample's";
    }
    return null;
  }

  /* A clean record of this sample holding `review`'s answers (types enforced, unknown fields dropped). */
  function normalize(review, data) {
    var clean = JSON.parse(JSON.stringify(data.blank));
    if (!review || typeof review !== "object") { return clean; }
    clean.reviewer = typeof review.reviewer === "string" ? review.reviewer : "";
    clean.date = typeof review.date === "string" ? review.date : "";
    var verdict = review.verdict && typeof review.verdict === "object" ? review.verdict : {};
    clean.verdict = { freeze: verdict.freeze === true, notes: typeof verdict.notes === "string" ? verdict.notes : "" };
    var byId = {};
    (Array.isArray(review.items) ? review.items : []).forEach(function (i) { if (i && typeof i === "object") { byId[i.id] = i; } });
    clean.items.forEach(function (item) {
      var a = byId[item.id];
      if (!a) { return; }
      var checks = a.checks && typeof a.checks === "object" ? a.checks : {};
      Object.keys(item.checks).forEach(function (c) { item.checks[c] = checks[c] === true; });
      item.notes = typeof a.notes === "string" ? a.notes : "";
    });
    return clean;
  }

  /* An item is reviewed when every check is ticked, or when it has a note (the note says what does not hold).
     This is read off the answers: nothing about it is stored, so the record's format does not change. */
  function itemDone(answer) {
    if (!answer || typeof answer !== "object") { return false; }
    if (String(answer.notes || "").trim()) { return true; }
    var checks = answer.checks && typeof answer.checks === "object" ? answer.checks : {};
    var keys = Object.keys(checks);
    return keys.length > 0 && keys.every(function (c) { return checks[c] === true; });
  }

  function progress(review) {
    return { done: review.items.filter(itemDone).length, total: review.items.length };
  }

  /* Where the draft of a sample lives in localStorage. Changing this would orphan saved drafts. */
  function storageKey(meta) { return "hcai-review:" + meta.corpus_version + ":" + meta.seed; }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      reviewMarkdown: reviewMarkdown, checkReview: checkReview, normalize: normalize,
      itemDone: itemDone, progress: progress, storageKey: storageKey
    };
  }
  if (typeof document === "undefined") { return; }

  function t(key) { return window.hcaiT ? window.hcaiT(key) : key; }
  function byId(id) { return document.getElementById(id); }

  function start() {
    var holder = byId("review-data");
    var form = byId("review");
    if (!holder || !form) { return; }
    var data = JSON.parse(holder.textContent);
    var draftKey = storageKey(data.meta);
    // The JSON text of the last export, to tell whether the answers have changed since. Beside the draft, not in it.
    var exportedKey = "hcai-review-exported:" + data.meta.corpus_version + ":" + data.meta.seed;
    var KEYS_KEY = "hcai-review-keys";  // "off" when the reviewer has turned the keyboard shortcuts off
    var statusEl = byId("review-status");
    var finishStatusEl = byId("finish-status");
    var bar = byId("review-bar");
    var finish = byId("finish");
    var articles = Array.prototype.slice.call(form.querySelectorAll("article[data-item]"));
    var chips = articles.map(function (a, n) { return bar.querySelector('a[data-chip="' + n + '"]'); });
    var state;

    function status(message) { statusEl.textContent = message; }
    function finishStatus(message) { finishStatusEl.textContent = message; }

    var staleKey = "";
    function loadDraft() {
      try {
        var raw = window.localStorage.getItem(draftKey);
        if (!raw) { return null; }
        var parsed;
        try { parsed = JSON.parse(raw); } catch (e) { parsed = null; }
        if (parsed && !checkReview(parsed, data)) { return normalize(parsed, data); }
        // another sample's draft (or an unreadable one): keep it under its own key instead of overwriting it
        staleKey = draftKey + ":set-aside:" + new Date().toISOString();
        window.localStorage.setItem(staleKey, raw);
        window.localStorage.removeItem(draftKey);
        return null;
      } catch (e) { return null; }
    }
    var saved = false;  // false: nothing saved in this visit yet; true: saved; null: this browser cannot save
    var savedAt = "";
    var hasDraft = false;
    function save() {
      try {
        window.localStorage.setItem(draftKey, JSON.stringify(state));
        if (!saved) { status(t("review.saved")); }  // announced once, not on every keystroke
        saved = true;
        hasDraft = true;
        savedAt = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      } catch (e) {
        if (saved !== null) { status(t("review.nosave")); }
        saved = null;
      }
      render();
    }
    function startingState() {
      return data.committed ? normalize(data.committed, data) : normalize(null, data);
    }

    var lastExport = null;
    try { lastExport = window.localStorage.getItem(exportedKey); } catch (e) { /* no storage: never "exported" */ }
    function markExported() {
      lastExport = JSON.stringify(state);
      try { window.localStorage.setItem(exportedKey, lastExport); } catch (e) { /* kept for this visit only */ }
      render();
    }

    function articleFor(id) {
      for (var i = 0; i < articles.length; i++) {
        if (articles[i].getAttribute("data-item") === id) { return articles[i]; }
      }
      return null;
    }

    function fill() {
      byId("reviewer").value = state.reviewer;
      byId("date").value = state.date;
      byId("verdict-freeze").checked = state.verdict.freeze;
      byId("verdict-notes").value = state.verdict.notes;
      state.items.forEach(function (answer) {
        var article = articleFor(answer.id);
        if (!article) { return; }
        Object.keys(answer.checks).forEach(function (c) {
          var box = article.querySelector('input[data-check="' + c + '"]');
          if (box) { box.checked = answer.checks[c]; }
        });
        article.querySelector("textarea[data-notes]").value = answer.notes;
      });
    }

    function read() {
      state.reviewer = byId("reviewer").value;
      state.date = byId("date").value;
      state.verdict.freeze = byId("verdict-freeze").checked;
      state.verdict.notes = byId("verdict-notes").value;
      state.items.forEach(function (answer) {
        var article = articleFor(answer.id);
        if (!article) { return; }
        Object.keys(answer.checks).forEach(function (c) {
          var box = article.querySelector('input[data-check="' + c + '"]');
          answer.checks[c] = !!(box && box.checked);
        });
        answer.notes = article.querySelector("textarea[data-notes]").value;
      });
    }

    /* Everything the page shows about the answers: progress, each item's state, and the save and export states. */
    function render() {
      var answers = {};
      state.items.forEach(function (a) { answers[a.id] = a; });
      articles.forEach(function (article, n) {
        var done = itemDone(answers[article.getAttribute("data-item")]);
        var label = t(done ? "review.item.done" : "review.item.todo");
        var badge = article.querySelector("[data-state]");
        if (badge) {
          badge.textContent = label;
          badge.classList.toggle("done", done);
        }
        if (chips[n]) {
          chips[n].classList.toggle("done", done);
          var hidden = chips[n].querySelector("[data-chip-state]");
          if (hidden) { hidden.textContent = ", " + label; }
        }
      });
      var p = progress(state);
      byId("review-progress").textContent = t("review.progress").replace("{done}", p.done).replace("{total}", p.total);
      var saveEl = byId("save-state");
      saveEl.classList.toggle("error", saved === null);
      saveEl.textContent = saved === null ? t("review.nosave.short")
        : savedAt ? t("review.saved.at") + " " + savedAt
        : hasDraft ? t("review.saved") : "";
      var now = JSON.stringify(state);
      byId("export-state").textContent =
        data.committed && now === JSON.stringify(normalize(data.committed, data)) ? t("review.export.committed")
        : lastExport === null ? t("review.export.none")
        : lastExport === now ? t("review.export.same") : t("review.export.changed");
      measure();
    }
    /* The strip's height, for what sticks or scrolls under it: the answers beside a text (--bar-h), and a jump
       to an item or a focused field (the scroll padding), which must not end up under the strip. */
    function measure() {
      document.documentElement.style.setProperty("--bar-h", bar.offsetHeight + "px");
      document.documentElement.style.scrollPaddingTop = bar.offsetHeight + "px";
    }

    function download(name, text, type) {
      var url = URL.createObjectURL(new Blob([text], { type: type }));
      var a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }
    var base = data.record.split("/").pop();
    function exportMarkdown() { read(); download(base + ".md", reviewMarkdown(data, state), "text/markdown;charset=utf-8"); }
    function exportJson() { read(); download(base + ".json", JSON.stringify(state, null, 2) + "\n", "application/json"); }

    var draft = loadDraft();
    var messages = [];
    if (draft) {
      state = draft;
      hasDraft = true;
    } else {
      state = startingState();
      if (data.committed) { messages.push(t("review.committed")); }
    }
    if (staleKey) { messages.push(t("review.stale") + " " + staleKey + ")."); }
    if (data.committed_mismatch) {
      messages.push(t("review.committed.mismatch") + ": " + data.committed_mismatch);
    }
    status(messages.join(" "));
    fill();

    function changed() { read(); save(); }
    function notAnAnswer(e) {
      var id = e.target && e.target.id;
      return id === "show-slots" || id === "import-file" || id === "keys-on";
    }
    form.addEventListener("submit", function (e) { e.preventDefault(); });
    form.addEventListener("input", function (e) { if (!notAnAnswer(e)) { changed(); } });
    form.addEventListener("change", function (e) { if (!notAnAnswer(e)) { changed(); } });

    /* "All hold": tick every check of one item. Per item only, never the whole page. */
    function tickAll(article) {
      var boxes = article.querySelectorAll("input[data-check]");
      for (var i = 0; i < boxes.length; i++) { boxes[i].checked = true; }
      changed();
    }
    form.addEventListener("click", function (e) {
      var button = e.target && e.target.closest ? e.target.closest("button[data-all]") : null;
      if (button) { tickAll(button.closest("article")); }
    });

    byId("date-today").addEventListener("click", function () {
      var d = new Date();
      function two(n) { return (n < 10 ? "0" : "") + n; }
      byId("date").value = d.getFullYear() + "-" + two(d.getMonth() + 1) + "-" + two(d.getDate());
      changed();
    });

    byId("export").addEventListener("click", function () {
      exportMarkdown();
      setTimeout(exportJson, 400);
      markExported();
      finishStatus(t("review.exported"));
    });
    byId("export-md").addEventListener("click", function () { exportMarkdown(); markExported(); });
    byId("export-json").addEventListener("click", function () { exportJson(); markExported(); });

    var slots = byId("show-slots");
    slots.addEventListener("change", function () { form.classList.toggle("slots-on", slots.checked); });
    form.classList.toggle("slots-on", slots.checked);  // a reload can keep the box ticked

    byId("clear").addEventListener("click", function () {
      if (!window.confirm(t("review.clear.confirm"))) { return; }
      try {
        window.localStorage.removeItem(draftKey);
        window.localStorage.removeItem(exportedKey);
      } catch (e) { /* nothing stored */ }
      state = startingState();
      saved = false;
      savedAt = "";
      hasDraft = false;
      lastExport = null;
      fill();
      render();
      status(data.committed ? t("review.committed") : "");
      finishStatus("");
    });

    var fileInput = byId("import-file");
    fileInput.addEventListener("change", function () {
      var file = fileInput.files && fileInput.files[0];
      if (!file) { return; }
      var reader = new FileReader();
      reader.onload = function () {
        var parsed;
        try { parsed = JSON.parse(String(reader.result)); } catch (e) {
          finishStatus(t("review.import.unreadable"));
          return;
        }
        var problem = checkReview(parsed, data);
        if (problem) {
          finishStatus(t("review.import.refused") + ": " + problem);
          return;
        }
        state = normalize(parsed, data);
        fill();
        save();
        markExported();  // these answers are the file that was just read
        finishStatus(t("review.imported"));
      };
      reader.readAsText(file);
      fileInput.value = "";
    });

    /* The current item: where the keys act. -1 before the first item, articles.length in "Verdict and export".
       It follows the scroll position, a jump (chip or key) and the focus. */
    var current = -1;
    var lockY = null;  // the scroll position right after a jump or a focus: the item stays current until a scroll
    function setCurrent(n) {
      if (n === current) { return; }
      if (articles[current]) {
        articles[current].classList.remove("current");
        if (chips[current]) { chips[current].removeAttribute("aria-current"); }
      }
      current = n;
      if (articles[n]) {
        articles[n].classList.add("current");
        var chip = chips[n];
        if (chip) {
          chip.setAttribute("aria-current", "true");
          var row = chip.closest("ol");
          if (row && row.scrollWidth > row.clientWidth) {  // a phone's one-row list: keep the chip in it
            var c = chip.getBoundingClientRect();
            var r = row.getBoundingClientRect();
            row.scrollLeft += c.left - r.left - (r.width - c.width) / 2;
          }
        }
      }
    }
    function go(n) {
      n = Math.max(0, Math.min(articles.length, n));
      var target = articles[n] || finish;
      setCurrent(n);
      if (articles[n]) { articles[n].focus({ preventScroll: true }); }
      target.scrollIntoView();
      lockY = window.scrollY;
    }
    function fromScroll() {
      if (lockY !== null && Math.abs(window.scrollY - lockY) < 2) { return; }
      lockY = null;
      var line = bar.offsetHeight + window.innerHeight * 0.3;
      var found = -1;
      for (var i = 0; i < articles.length; i++) {
        if (articles[i].getBoundingClientRect().top > line) { break; }
        found = i;
      }
      if (finish.getBoundingClientRect().top <= line) { found = articles.length; }
      setCurrent(found);
    }
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) { return; }
      ticking = true;
      window.requestAnimationFrame(function () { ticking = false; fromScroll(); });
    }, { passive: true });
    window.addEventListener("resize", measure);
    form.addEventListener("focusin", function (e) {
      var article = e.target && e.target.closest ? e.target.closest("article[data-item]") : null;
      var n = article ? articles.indexOf(article) : -1;
      if (n >= 0) {
        setCurrent(n);
        lockY = window.scrollY;
      }
    });
    bar.addEventListener("click", function (e) {
      var chip = e.target && e.target.closest ? e.target.closest("a[data-chip]") : null;
      if (!chip) { return; }
      e.preventDefault();
      go(Number(chip.getAttribute("data-chip")));
      try { window.history.replaceState(null, "", chip.getAttribute("href")); } catch (err) { /* opened from disk */ }
    });

    /* Keyboard shortcuts: single keys, only while the focus is not in a field, and they can be turned off. */
    var keysBox = byId("keys-on");
    var keysOn = true;
    try { keysOn = window.localStorage.getItem(KEYS_KEY) !== "off"; } catch (e) { /* on */ }
    keysBox.checked = keysOn;
    keysBox.addEventListener("change", function () {
      keysOn = keysBox.checked;
      try {
        if (keysOn) { window.localStorage.removeItem(KEYS_KEY); } else { window.localStorage.setItem(KEYS_KEY, "off"); }
      } catch (e) { /* for this visit only */ }
    });
    document.addEventListener("keydown", function (e) {
      if (!keysOn || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) { return; }
      var el = e.target;
      var tag = el && el.tagName;
      var typing = tag === "TEXTAREA" || tag === "SELECT" || (el && el.isContentEditable) ||
        (tag === "INPUT" && !/^(checkbox|radio|button|file)$/.test(el.type));
      if (e.key === "Escape") {
        if (typing) {
          el.blur();
          if (articles[current]) { articles[current].focus({ preventScroll: true }); }
        }
        return;
      }
      if (typing) { return; }
      var key = e.key.toLowerCase();
      var article = articles[current];
      if (key === "j") {
        go(current + 1);
      } else if (key === "k") {
        go(current - 1);
      } else if (!article) {
        return;
      } else if (/^[1-9]$/.test(key)) {
        var box = article.querySelectorAll("input[data-check]")[Number(key) - 1];
        if (!box) { return; }
        box.checked = !box.checked;
        changed();
      } else if (key === "a") {
        tickAll(article);
      } else if (key === "n") {
        article.querySelector("textarea[data-notes]").focus();
      } else {
        return;
      }
      e.preventDefault();
    });

    document.addEventListener("hcai:lang", render);
    render();
    var anchored = /^#item-(\d+)$/.exec(window.location.hash);
    if (anchored && articles[Number(anchored[1])]) {
      setCurrent(Number(anchored[1]));
      lockY = window.scrollY;
    } else {
      fromScroll();
    }
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
