/* Realism review form (ADR 0010). Answers are a draft in localStorage; "Export review" downloads the record
   (Markdown + JSON) to commit to docs/pilot/. reviewMarkdown mirrors hcai.review.review_markdown line by line,
   and checkReview mirrors hcai.review.check_review; a test runs both sides on the same answers. */
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

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { reviewMarkdown: reviewMarkdown, checkReview: checkReview, normalize: normalize };
  }
  if (typeof document === "undefined") { return; }

  function t(key) { return window.hcaiT ? window.hcaiT(key) : key; }

  document.addEventListener("DOMContentLoaded", function () {
    var holder = document.getElementById("review-data");
    var form = document.getElementById("review");
    if (!holder || !form) { return; }
    var data = JSON.parse(holder.textContent);
    var storageKey = "hcai-review:" + data.meta.corpus_version + ":" + data.meta.seed;
    var statusEl = document.getElementById("review-status");
    var state;

    function status(message) { statusEl.textContent = message; }

    function loadDraft() {
      try {
        var raw = window.localStorage.getItem(storageKey);
        if (!raw) { return null; }
        var parsed = JSON.parse(raw);
        return checkReview(parsed, data) ? null : normalize(parsed, data);
      } catch (e) { return null; }
    }
    function save() {
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(state));
        status(t("review.saved"));
      } catch (e) {
        status(t("review.nosave"));
      }
    }

    function articleFor(id) {
      var articles = form.querySelectorAll("article[data-item]");
      for (var i = 0; i < articles.length; i++) {
        if (articles[i].getAttribute("data-item") === id) { return articles[i]; }
      }
      return null;
    }

    function fill() {
      document.getElementById("reviewer").value = state.reviewer;
      document.getElementById("date").value = state.date;
      document.getElementById("verdict-freeze").checked = state.verdict.freeze;
      document.getElementById("verdict-notes").value = state.verdict.notes;
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
      state.reviewer = document.getElementById("reviewer").value;
      state.date = document.getElementById("date").value;
      state.verdict.freeze = document.getElementById("verdict-freeze").checked;
      state.verdict.notes = document.getElementById("verdict-notes").value;
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
    if (draft) {
      state = draft;
    } else if (data.committed) {
      state = normalize(data.committed, data);
      status(t("review.committed"));
    } else {
      state = normalize(null, data);
    }
    if (data.committed_mismatch) {
      status(t("review.committed.mismatch") + ": " + data.committed_mismatch);
    }
    fill();

    form.addEventListener("submit", function (e) { e.preventDefault(); });
    form.addEventListener("input", function (e) {
      if (e.target && (e.target.id === "show-slots" || e.target.id === "import-file")) { return; }
      read();
      save();
    });
    form.addEventListener("change", function (e) {
      if (e.target && (e.target.id === "show-slots" || e.target.id === "import-file")) { return; }
      read();
      save();
    });

    document.getElementById("export").addEventListener("click", function () {
      exportMarkdown();
      setTimeout(exportJson, 400);
      status(t("review.exported"));
    });
    document.getElementById("export-md").addEventListener("click", exportMarkdown);
    document.getElementById("export-json").addEventListener("click", exportJson);

    var slots = document.getElementById("show-slots");
    slots.addEventListener("change", function () { form.classList.toggle("slots-on", slots.checked); });

    document.getElementById("clear").addEventListener("click", function () {
      if (!window.confirm(t("review.clear.confirm"))) { return; }
      try { window.localStorage.removeItem(storageKey); } catch (e) { /* nothing stored */ }
      state = normalize(null, data);
      fill();
      status("");
    });

    var fileInput = document.getElementById("import-file");
    fileInput.addEventListener("change", function () {
      var file = fileInput.files && fileInput.files[0];
      if (!file) { return; }
      var reader = new FileReader();
      reader.onload = function () {
        var parsed;
        try { parsed = JSON.parse(String(reader.result)); } catch (e) {
          status(t("review.import.unreadable"));
          return;
        }
        var problem = checkReview(parsed, data);
        if (problem) {
          status(t("review.import.refused") + ": " + problem);
          return;
        }
        state = normalize(parsed, data);
        fill();
        save();
        status(t("review.imported"));
      };
      reader.readAsText(file);
      fileInput.value = "";
    });
  });
})();
