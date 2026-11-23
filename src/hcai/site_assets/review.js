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

  /* What to say about the export, as an interface string key. `now` and `committed` are the JSON text of the
     answers and of the committed review (null when there is none); `last` says which answers each of the two
     files was last written with ({ md: text, json: text }; a file never written is missing), or is null. */
  function exportState(now, committed, last) {
    if (committed !== null && now === committed) { return "review.export.committed"; }
    var md = !!last && last.md === now;
    var json = !!last && last.json === now;
    if (md && json) { return "review.export.same"; }
    if (md) { return "review.export.only.md"; }
    if (json) { return "review.export.only.json"; }
    return last && (last.md || last.json) ? "review.export.changed" : "review.export.none";
  }

  /* The export marker as stored (see exportState's `last`), or null when there is none or it is not one. */
  function parseExported(raw) {
    var parsed;
    try { parsed = JSON.parse(raw); } catch (e) { return null; }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) { return null; }
    var out = {};
    if (typeof parsed.md === "string") { out.md = parsed.md; }
    if (typeof parsed.json === "string") { out.json = parsed.json; }
    return out.md || out.json ? out : null;
  }

  /* The current item from where things are on screen (tops from the top of the window): the last item whose top
     has reached `line`; -1 before the first; tops.length once "Verdict and export" has reached the line, or is
     on screen at the end of the page (on a tall screen it never gets as high as the line). */
  function currentIndex(tops, finishTop, line, atEnd, height) {
    var found = -1;
    for (var i = 0; i < tops.length; i++) {
      if (tops[i] > line) { break; }
      found = i;
    }
    if (finishTop <= line || (atEnd && finishTop < height)) { found = tops.length; }
    return found;
  }

  /* What a key does: { act: "next" | "prev" | "all" | "notes" | "leave" }, { act: "check", n: <from 0> }, or
     null. While the reviewer types in a field only Escape does anything. */
  function keyAction(key, typing) {
    if (key === "Escape") { return typing ? { act: "leave" } : null; }
    if (typing) { return null; }
    var k = String(key).toLowerCase();
    if (k === "j") { return { act: "next" }; }
    if (k === "k") { return { act: "prev" }; }
    if (/^[1-9]$/.test(k)) { return { act: "check", n: Number(k) - 1 }; }
    if (k === "a") { return { act: "all" }; }
    if (k === "n") { return { act: "notes" }; }
    return null;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      reviewMarkdown: reviewMarkdown, checkReview: checkReview, normalize: normalize,
      itemDone: itemDone, progress: progress, storageKey: storageKey,
      exportState: exportState, parseExported: parseExported, currentIndex: currentIndex, keyAction: keyAction
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
    // Which answers each exported file was last written with (exportState), to tell whether they have changed
    // since. Kept beside the draft, not in it.
    var exportedKey = "hcai-review-exported:" + data.meta.corpus_version + ":" + data.meta.seed;
    var KEYS_KEY = "hcai-review-keys";  // "off" when the reviewer has turned the keyboard shortcuts off
    var root = document.documentElement;
    var statusEl = byId("review-status");
    var finishStatusEl = byId("finish-status");
    var saidEl = byId("review-said");
    var bar = byId("review-bar");
    var finish = byId("finish");
    var articles = Array.prototype.slice.call(form.querySelectorAll("article[data-item]"));
    var chips = articles.map(function (a, n) { return bar.querySelector('a[data-chip="' + n + '"]'); });
    var committedText = data.committed ? JSON.stringify(normalize(data.committed, data)) : null;
    var state;
    var current = -1;  // the current item: see "The current item" below

    function status(message) { statusEl.textContent = message; }
    function finishStatus(message) { finishStatusEl.textContent = message; }
    function say(message) { saidEl.textContent = message; }  // for a screen reader: what a key or "All hold" did

    var exported = null;
    try { exported = parseExported(window.localStorage.getItem(exportedKey)); } catch (e) { /* no storage */ }
    function markExported(files) {
      var now = JSON.stringify(state);
      exported = exported || {};
      files.forEach(function (file) { exported[file] = now; });
      try { window.localStorage.setItem(exportedKey, JSON.stringify(exported)); } catch (e) { /* this visit only */ }
      render();
    }
    function forgetExported() {
      exported = null;
      try { window.localStorage.removeItem(exportedKey); } catch (e) { /* nothing stored */ }
    }

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

    function itemName(n) { return t("review.item") + " " + (n + 1); }

    /* The strip's second line: the current item and how many of its checks are ticked. It is what shows a key's
       effect when the checklist itself is not on screen (below the text, on a narrow window). */
    function renderCurrent() {
      var article = articles[current];
      var text = "";
      if (article) {
        var boxes = article.querySelectorAll("input[data-check]");
        var ticked = 0;
        for (var i = 0; i < boxes.length; i++) { if (boxes[i].checked) { ticked++; } }
        text = t("review.current").replace("{n}", current + 1).replace("{ticked}", ticked).replace("{total}", boxes.length);
      }
      byId("review-current").textContent = text;
    }

    /* Everything the page shows about the answers: progress, each item's state, and the save and export states. */
    function render() {
      var answers = {};
      state.items.forEach(function (a) { answers[a.id] = a; });
      articles.forEach(function (article, n) {
        var done = itemDone(answers[article.getAttribute("data-item")]);
        var label = done ? t("review.item.done") : t("review.item.todo");
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
      renderCurrent();
      var saveEl = byId("save-state");
      saveEl.classList.toggle("error", saved === null);
      saveEl.textContent = saved === null ? t("review.nosave.short")
        : savedAt ? t("review.saved.at") + " " + savedAt
        : hasDraft ? t("review.saved") : "";
      byId("export-state").textContent = t(exportState(JSON.stringify(state), committedText, exported));
      measure();
    }
    /* The height the strip covers while it sticks (none on a short screen, where it scrolls away), for what sticks
       or scrolls under it: the answers beside a text (--bar-h), and a jump to an item or a focused field (the
       scroll padding), which must not end up under the strip. */
    function stuck() { return window.getComputedStyle(bar).position === "sticky" ? bar.offsetHeight : 0; }
    function measure() {
      var height = stuck() + "px";
      root.style.setProperty("--bar-h", height);
      root.style.scrollPaddingTop = height;
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
    function exportMarkdown() {
      read();
      download(base + ".md", reviewMarkdown(data, state), "text/markdown;charset=utf-8");
      markExported(["md"]);
    }
    function exportJson() {
      read();
      download(base + ".json", JSON.stringify(state, null, 2) + "\n", "application/json");
      markExported(["json"]);
    }

    var draft = loadDraft();
    var messages = [];
    if (draft) {
      state = draft;
      hasDraft = true;
    } else {
      state = startingState();
      forgetExported();  // the marker is about a draft: without one (none, or set aside) it would mislead
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
      say(itemName(articles.indexOf(article)) + ": " + t("review.said.all"));
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

    // Each file is marked as holding the answers it was written with: "Markdown only" leaves the JSON stale.
    byId("export").addEventListener("click", function () {
      exportMarkdown();
      setTimeout(exportJson, 400);
      markExported(["json"]);  // on its way: the state does not flicker through "only the Markdown"
      finishStatus(t("review.exported"));
    });
    byId("export-md").addEventListener("click", exportMarkdown);
    byId("export-json").addEventListener("click", exportJson);

    var slots = byId("show-slots");
    slots.addEventListener("change", function () { form.classList.toggle("slots-on", slots.checked); });
    form.classList.toggle("slots-on", slots.checked);  // a reload can keep the box ticked

    byId("clear").addEventListener("click", function () {
      if (!window.confirm(t("review.clear.confirm"))) { return; }
      try { window.localStorage.removeItem(draftKey); } catch (e) { /* nothing stored */ }
      forgetExported();
      state = startingState();
      saved = false;
      savedAt = "";
      hasDraft = false;
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
        exported = null;  // these answers are in the JSON file that was just read; its Markdown is not known
        save();
        markExported(["json"]);
        finishStatus(t("review.imported"));
      };
      reader.readAsText(file);
      fileInput.value = "";
    });

    /* The current item: where the keys act. -1 before the first item, articles.length in "Verdict and export".
       It follows the scroll position (currentIndex). An item the reader is sent to (a jump link, a key, the
       focus) is current until the next scroll (lockY), but only while it is on screen: a page that comes back at
       another scroll position (a reload, the Back button) must never leave the keys acting out of sight. */
    var lockY = null;
    function setCurrent(n) {
      if (n === current) { return; }
      if (chips[current]) { chips[current].removeAttribute("aria-current"); }
      if (articles[current]) { articles[current].classList.remove("current"); }
      current = n;
      if (articles[n]) {
        articles[n].classList.add("current");
        var chip = chips[n];
        if (chip) {
          chip.setAttribute("aria-current", "true");
          var row = chip.closest("ol");
          if (row && row.scrollWidth > row.clientWidth) {  // a one-row list: keep the chip in it
            var c = chip.getBoundingClientRect();
            var r = row.getBoundingClientRect();
            row.scrollLeft += c.left - r.left - (r.width - c.width) / 2;
          }
        }
      }
      renderCurrent();
    }
    function onScreen(el, itsTop) {
      var r = el.getBoundingClientRect();
      return itsTop ? r.top >= stuck() - 4 && r.top < window.innerHeight : r.bottom > stuck() && r.top < window.innerHeight;
    }
    function claim(n, itsTop) {
      if (!onScreen(articles[n] || finish, itsTop)) { return false; }
      setCurrent(n);
      lockY = window.scrollY;
      return true;
    }
    function go(n) {
      n = Math.max(0, Math.min(articles.length, n));
      var target = articles[n] || finish;
      target.focus({ preventScroll: true });  // so that Tab goes on from there
      target.scrollIntoView();
      claim(n, false);
    }
    function fromScroll() {
      if (lockY !== null && Math.abs(window.scrollY - lockY) < 2) { return; }
      lockY = null;
      var atEnd = window.innerHeight + window.scrollY >= root.scrollHeight - 2;
      setCurrent(currentIndex(
        articles.map(function (a) { return a.getBoundingClientRect().top; }), finish.getBoundingClientRect().top,
        stuck() + window.innerHeight * 0.3, atEnd, window.innerHeight));
    }
    /* A link to an item (#item-N, #finish): that item is current if the page did go there. */
    function fromHash() {
      var hash = window.location.hash;
      var item = /^#item-(\d+)$/.exec(hash);
      var n = item ? Number(item[1]) : hash === "#finish" ? articles.length : -1;
      if (n < 0 || n > articles.length || !claim(n, true)) {
        lockY = null;
        fromScroll();
      }
    }
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) { return; }
      ticking = true;
      window.requestAnimationFrame(function () { ticking = false; fromScroll(); });
    }, { passive: true });
    window.addEventListener("resize", function () { measure(); fromScroll(); });
    window.addEventListener("hashchange", fromHash);
    window.addEventListener("load", fromHash);
    window.addEventListener("pageshow", fromHash);
    form.addEventListener("focusin", function (e) {
      var article = e.target && e.target.closest ? e.target.closest("article[data-item]") : null;
      var n = article ? articles.indexOf(article) : -1;
      if (n >= 0) { claim(n, false); }
    });
    bar.addEventListener("click", function (e) {
      var chip = e.target && e.target.closest ? e.target.closest("a.chip") : null;
      if (!chip) { return; }
      e.preventDefault();
      go(chip.hasAttribute("data-chip") ? Number(chip.getAttribute("data-chip")) : articles.length);
      try { window.history.replaceState(null, "", chip.getAttribute("href")); } catch (err) { /* opened from disk */ }
    });

    /* Keyboard shortcuts (keyAction): single keys, only while the focus is not in a field, and they can be
       turned off. They only ever change an item that is on screen. */
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
      var typing = tag === "TEXTAREA" || tag === "SELECT" || !!(el && el.isContentEditable) ||
        (tag === "INPUT" && !/^(checkbox|radio|button|file)$/.test(el.type));
      var action = keyAction(e.key, typing);
      if (!action) { return; }
      var article = articles[current];
      if (action.act === "leave") {
        el.blur();
        if (article) { article.focus({ preventScroll: true }); }
        return;
      }
      if (action.act === "next") {
        go(current + 1);
      } else if (action.act === "prev") {
        go(current - 1);
      } else if (!article || !onScreen(article, false)) {
        return;
      } else if (action.act === "check") {
        var box = article.querySelectorAll("input[data-check]")[action.n];
        if (!box) { return; }
        box.checked = !box.checked;
        changed();
        say(itemName(current) + ": " + box.parentNode.textContent.trim() + ", " +
          (box.checked ? t("review.said.ticked") : t("review.said.unticked")));
      } else if (action.act === "all") {
        tickAll(article);
      } else {
        article.querySelector("textarea[data-notes]").focus();
      }
      e.preventDefault();
    });

    document.addEventListener("hcai:lang", render);
    render();
    fromHash();
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
