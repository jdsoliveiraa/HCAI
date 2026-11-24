/* Human audit of the judge (RQ4): each rater labels every quote faithful or unfaithful. The labels are a draft in
   localStorage (one per sample); "Export labels" downloads labels_<rater>.csv, one row per item x criterion
   (item_id, criterion, label, comment), which `hcai.audit.read_criterion_labels` reads. The page never shows the
   judge's verdict; the item label shown is derived from the rater's own quote labels (unfaithful if any is). */
(function () {
  "use strict";
  var root = document.getElementById("audit");
  if (!root) { return; }
  var KEY = "hcai-audit:" + root.getAttribute("data-sample");
  var items = Array.prototype.slice.call(root.querySelectorAll("article.audit-item"));
  var status = document.getElementById("audit-status");
  var rater = document.getElementById("rater");
  var canSave = true;

  function t(key) { return window.hcaiT ? window.hcaiT(key) : key; }

  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : { rater: "", labels: {}, comments: {} };
    } catch (e) {
      canSave = false;
      return { rater: "", labels: {}, comments: {} };
    }
  }
  var state = load();
  state.labels = state.labels || {};
  state.comments = state.comments || {};

  function save() {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(state));
      say(t("audit.saved"));
    } catch (e) {
      canSave = false;
      say(t("audit.nosave"));
    }
  }
  function say(text) { if (status) { status.textContent = text; } }

  function itemLabel(article) {
    var id = article.getAttribute("data-item");
    var n = Number(article.getAttribute("data-criteria"));
    var any = false;
    for (var k = 1; k <= n; k++) {
      var label = state.labels[id + "-" + k];
      if (!label) { return ""; }
      if (label === "unfaithful") { any = true; }
    }
    return any ? "unfaithful" : "faithful";
  }

  function render() {
    var done = 0;
    items.forEach(function (article) {
      var label = itemLabel(article);
      if (label) { done += 1; }
      var mark = article.querySelector("[data-item-label]");
      if (mark) { mark.textContent = label ? "· " + t("audit.item.label") + ": " + t("audit." + label) : ""; }
    });
    var progress = document.getElementById("audit-progress");
    if (progress) {
      progress.textContent = t("audit.progress").replace("{done}", done).replace("{total}", items.length);
    }
    return done;
  }

  /* restore the draft */
  rater.value = state.rater || "";
  Array.prototype.forEach.call(root.querySelectorAll('input[type="radio"]'), function (input) {
    var key = input.getAttribute("data-item") + "-" + input.getAttribute("data-criterion");
    input.checked = state.labels[key] === input.value;
    input.addEventListener("change", function () {
      if (input.checked) { state.labels[key] = input.value; save(); render(); }
    });
  });
  Array.prototype.forEach.call(root.querySelectorAll("textarea[data-comment]"), function (area) {
    var id = area.getAttribute("data-comment");
    area.value = state.comments[id] || "";
    area.addEventListener("input", function () { state.comments[id] = area.value; save(); });
  });
  rater.addEventListener("input", function () { state.rater = rater.value; save(); });

  function cell(value) {
    var text = String(value === undefined || value === null ? "" : value);
    return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function sheet() {
    var lines = ["item_id,criterion,label,comment"];
    items.forEach(function (article) {
      var id = article.getAttribute("data-item");
      var n = Number(article.getAttribute("data-criteria"));
      for (var k = 1; k <= n; k++) {
        lines.push([id, k, state.labels[id + "-" + k] || "", k === 1 ? (state.comments[id] || "") : ""].map(cell).join(","));
      }
    });
    return lines.join("\r\n") + "\r\n";
  }

  function slug(name) {
    var s = name.normalize ? name.normalize("NFD").replace(/[̀-ͯ]/g, "") : name;
    return s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  }

  document.getElementById("audit-export").addEventListener("click", function () {
    var name = slug(rater.value || "");
    if (!name) { say(t("audit.need.name")); rater.focus(); return; }
    if (render() < items.length && !window.confirm(t("audit.incomplete"))) { return; }
    /* a byte-order mark, so a spreadsheet opens the accented text correctly */
    var url = URL.createObjectURL(new Blob(["﻿" + sheet()], { type: "text/csv;charset=utf-8" }));
    var a = document.createElement("a");
    a.href = url;
    a.download = "labels_" + name + ".csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    say(t("audit.exported"));
  });

  document.getElementById("audit-clear").addEventListener("click", function () {
    if (!window.confirm(t("audit.clear.confirm"))) { return; }
    state = { rater: rater.value, labels: {}, comments: {} };
    Array.prototype.forEach.call(root.querySelectorAll('input[type="radio"]'), function (i) { i.checked = false; });
    Array.prototype.forEach.call(root.querySelectorAll("textarea[data-comment]"), function (a) { a.value = ""; });
    save();
    render();
  });

  document.addEventListener("hcai:lang", render);
  render();
  if (!canSave) { say(t("audit.nosave")); }
}());
